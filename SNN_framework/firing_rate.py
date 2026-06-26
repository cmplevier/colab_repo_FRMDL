from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from .neurons import LIFNeuron


class FiringRateTracker:
    """Track per-layer and overall LIF firing rates for LIFNeuron modules.

    Stats are preserved after the context manager exits.
    """

    def __init__(self, model):
        self._hooks = []
        self._stats: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {"spikes": 0.0, "numel": 0, "calls": 0}
        )

        for name, module in model.named_modules():
            if isinstance(module, LIFNeuron):
                handle = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(handle)

    def _make_hook(self, name: str):
        @torch._dynamo.disable
        def hook(module: LIFNeuron, inputs: tuple[Any, ...], output: torch.Tensor):
            if output is None:
                return

            if isinstance(output, (tuple, list)):
                output = output[0]

            if not torch.is_tensor(output):
                return

            out = output.detach()

            self._stats[name]["spikes"] += float(out.sum().item())
            self._stats[name]["numel"] += int(out.numel())
            self._stats[name]["calls"] += 1

        return hook

    def firing_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}

        total_spikes = 0.0
        total_numel = 0

        for name, stats in self._stats.items():
            spikes = float(stats["spikes"])
            numel = int(stats["numel"])

            rates[name] = spikes / numel if numel > 0 else 0.0

            total_spikes += spikes
            total_numel += numel

        rates["overall"] = total_spikes / total_numel if total_numel > 0 else 0.0
        return rates

    def summary(self) -> dict[str, dict[str, float | int]]:
        result = {}

        for name, stats in self._stats.items():
            spikes = float(stats["spikes"])
            numel = int(stats["numel"])
            calls = int(stats["calls"])

            result[name] = {
                "firing_rate": spikes / numel if numel > 0 else 0.0,
                "spikes": spikes,
                "numel": numel,
                "calls": calls,
            }

        return result

    def num_tracked_layers(self) -> int:
        return len(self._hooks)

    def reset(self) -> None:
        self._stats.clear()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def remove(self) -> None:
        self.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()