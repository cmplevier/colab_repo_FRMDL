from __future__ import annotations

import torch

from .neurons import LIFNeuron
from .EMSblock import SnnConv2d


class EnergyTracker:
    """
    Estimates per-layer and total SNN energy using the EMS-YOLO paper formula:

        E_l = fr_l * T * Connections_l * E_AC

    Usage:
        with FiringRateTracker(model) as fr_tracker:
            energy_tracker = EnergyTracker(model, T=cfg["model"]["T"])
            evaluate(model, ...)                  # triggers hooks
            fr    = fr_tracker.firing_rates()
            stats = energy_tracker.energy(fr)
            energy_tracker.remove()
    """

    E_AC:  float = 0.9e-12   # joules — accumulate at 45nm CMOS
    E_MAC: float = 4.6e-12   # joules — multiply-accumulate at 45nm CMOS

    def __init__(self, model, T: int) -> None:
        self.T = T
        # ordered list of (lif_name, conv_name) pairs discovered by DFS traversal
        self._lif_conv_pairs: list[tuple[str, str]] = []
        # conv_name -> connections per image per timestep (populated on first forward)
        self._conv_connections: dict[str, int] = {}
        self._hooks: list = []

        self._build_pairs(model)
        self._register_hooks(model)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_pairs(self, model) -> None:
        """
         when a SnnConv2d is encountered, pair it with the most recently seen LIFNeuron. 
         This captures both the main path pairs and the shortcut path pairs.
        """
        last_lif: str | None = None
        for name, module in model.named_modules():
            if isinstance(module, LIFNeuron):
                last_lif = name
            elif isinstance(module, SnnConv2d) and last_lif is not None:
                self._lif_conv_pairs.append((last_lif, name))

    def _register_hooks(self, model) -> None:
        conv_map = {n: m for n, m in model.named_modules() if isinstance(m, SnnConv2d)}
        for _, conv_name in self._lif_conv_pairs:
            module = conv_map[conv_name]
            h = module.register_forward_hook(self._make_conv_hook(conv_name))
            self._hooks.append(h)

    def _make_conv_hook(self, name: str):
        @torch._dynamo.disable
        def hook(module: SnnConv2d, inputs, output):
            # Record connection count once — shape is constant across batches
            if name in self._conv_connections:
                return
            # output: [T, B, C_out, H_out, W_out]
            C_out = output.shape[2]
            H_out = output.shape[3]
            W_out = output.shape[4]
            k = module.conv.kernel_size
            Kh, Kw = k if isinstance(k, tuple) else (k, k)
            C_in = module.conv.in_channels
            self._conv_connections[name] = int(Kh * Kw * C_in * C_out * H_out * W_out)
        return hook

    # ------------------------------------------------------------------
    # Energy calculation
    # ------------------------------------------------------------------

    def energy(self, firing_rates: dict[str, float]) -> dict:
        """
        Compute energy estimates given per-layer firing rates from FiringRateTracker.

        Args:
            firing_rates: dict returned by FiringRateTracker.firing_rates(),
                          keyed by LIFNeuron module name.

        Returns:
            {
                "E_SNN_J":   total SNN energy in joules (per image),
                "E_ANN_J":   equivalent ANN energy in joules (per image, simulated),
                "ratio":     E_SNN_J / E_ANN_J  (lower = more efficient; None if ANN=0),
                "per_layer": {
                    conv_name: {
                        "lif_layer":   str,
                        "fr":          float,
                        "connections": int,
                        "E_SNN_J":     float,
                        "E_ANN_J":     float,
                    }
                }
            }
        """
        total_snn = 0.0
        total_ann = 0.0
        per_layer: dict[str, dict] = {}

        for lif_name, conv_name in self._lif_conv_pairs:
            if conv_name not in self._conv_connections:
                # Hook never fired — model not forwarded yet, or layer not reached
                continue
            fr    = firing_rates.get(lif_name, 0.0)
            conns = self._conv_connections[conv_name]
            e_snn = fr * self.T * conns * self.E_AC
            e_ann = conns * self.E_MAC  # ANN simulation
            total_snn += e_snn
            total_ann += e_ann
            per_layer[conv_name] = {
                "lif_layer":   lif_name,
                "fr":          fr,
                "connections": conns,
                "E_SNN_J":     e_snn,
                "E_ANN_J":     e_ann,
            }

        ratio = total_snn / total_ann if total_ann > 0.0 else None
        return {
            "E_SNN_J":   total_snn,
            "E_ANN_J":   total_ann,
            "ratio":     ratio,
            "per_layer": per_layer,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def num_tracked_pairs(self) -> int:
        """Number of (LIF, SnnConv2d) pairs discovered in the model."""
        return len(self._lif_conv_pairs)

    def num_resolved_pairs(self) -> int:
        """Number of pairs whose connection count has been populated by a forward pass."""
        return len(self._conv_connections)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()
