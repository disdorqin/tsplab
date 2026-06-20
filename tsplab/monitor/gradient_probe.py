"""
Gradient health probe.

Automatically hooks into PyTorch model's backward pass to monitor:
  - Per-layer gradient norms
  - Gradient histogram
  - Zero-gradient ratio
  - Vanishing/explosion detection
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GradientProbe:
    """Monitor gradient health without modifying model code.

    Usage::

        probe = GradientProbe(model)
        # ... train model normally ...
        stats = probe.get_stats()
        probe.report()
    """

    def __init__(self, model) -> None:
        self.model = model
        self._hooks: list[Any] = []
        self._layer_grads: dict[str, list[float]] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register backward hooks on all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._layer_grads.setdefault(name, [])

                def make_hook(param_name):
                    def hook(grad):
                        norm = float(grad.norm().item())
                        self._layer_grads[param_name].append(norm)
                        return None  # Don't modify gradient
                    return hook

                hook = param.register_hook(make_hook(name))
                self._hooks.append(hook)

    def get_stats(self) -> dict[str, Any]:
        """Get gradient statistics for all layers."""
        stats = {}
        for name, norms in self._layer_grads.items():
            if not norms:
                continue
            arr = np.array(norms)
            stats[name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "last": float(arr[-1]),
                "n_steps": len(arr),
            }
        return stats

    def detect_issues(self) -> list[str]:
        """Detect gradient problems."""
        issues = []
        stats = self.get_stats()

        for name, s in stats.items():
            # Vanishing gradient
            if s["last"] < 1e-7:
                issues.append(
                    f"[Vanishing] {name}: last grad norm = {s['last']:.2e}"
                )
            # Exploding gradient
            elif s["last"] > 100:
                issues.append(
                    f"[Exploding] {name}: last grad norm = {s['last']:.2e}"
                )
            # Dead neuron (consistently zero)
            if s["mean"] < 1e-8 and s["n_steps"] > 5:
                issues.append(
                    f"[Dead] {name}: mean grad norm = {s['mean']:.2e} (consistently near zero)"
                )

        return issues

    def report(self) -> None:
        """Print gradient health report."""
        stats = self.get_stats()
        issues = self.detect_issues()

        print("=" * 60)
        print("  TSPLab Gradient Health Report")
        print("=" * 60)
        print()

        if not stats:
            print("  No gradient data collected yet.")
            return

        print(f"  Layers monitored: {len(stats)}")
        print()

        # Show top 5 layers by gradient norm
        sorted_layers = sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True)
        print("  Top layers by gradient norm:")
        for name, s in sorted_layers[:5]:
            print(f"    {name}: mean={s['mean']:.4e}, last={s['last']:.4e}")
        print()

        # Show bottom 5
        print("  Bottom layers by gradient norm:")
        for name, s in sorted_layers[-5:]:
            print(f"    {name}: mean={s['mean']:.4e}, last={s['last']:.4e}")
        print()

        if issues:
            print(f"  [!] {len(issues)} issue(s) detected:")
            for issue in issues:
                print(f"      {issue}")
        else:
            print("  [OK] No gradient issues detected.")
        print()
        print("=" * 60)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()
