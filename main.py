"""Monte Carlo simulation for mechanical load vs material strength."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Input parameters for a simulation run."""

    material_strength_mean: float = 100.0
    material_strength_std: float = 10.0
    load_mean: float = 90.0
    load_std: float = 15.0
    num_simulations: int = 10_000
    seed: Optional[int] = 42

    def validate(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if self.material_strength_std <= 0:
            raise ValueError("material_strength_std must be > 0")
        if self.load_std <= 0:
            raise ValueError("load_std must be > 0")


@dataclass(frozen=True)
class SimulationResult:
    """Computed arrays and aggregate statistics for a simulation run."""

    material_strengths: np.ndarray
    loads: np.ndarray
    failures: np.ndarray

    @property
    def failure_rate(self) -> float:
        return float(np.mean(self.failures))

    @property
    def failure_count(self) -> int:
        return int(np.sum(self.failures))


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run Monte Carlo simulation and return full output arrays."""
    config.validate()
    rng = np.random.default_rng(config.seed)

    material_strengths = rng.normal(
        loc=config.material_strength_mean,
        scale=config.material_strength_std,
        size=config.num_simulations,
    )
    loads = rng.normal(
        loc=config.load_mean,
        scale=config.load_std,
        size=config.num_simulations,
    )

    failures = loads > material_strengths
    return SimulationResult(
        material_strengths=material_strengths,
        loads=loads,
        failures=failures,
    )


def print_summary(result: SimulationResult) -> None:
    print(f"Failure Rate: {result.failure_rate * 100:.2f}%")
    print(f"Failure Count: {result.failure_count}/{result.failures.size}")


def plot_simulation(result: SimulationResult, config: SimulationConfig) -> None:
    """Plot histograms for material strength and load samples."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(result.material_strengths, bins=30, alpha=0.7, label="Material Strength (kg)")
    plt.hist(result.loads, bins=30, alpha=0.7, label="Applied Load (kg)")
    plt.axvline(
        config.material_strength_mean,
        color="tab:blue",
        linestyle="--",
        label="Mean Material Strength",
    )
    plt.axvline(config.load_mean, color="tab:orange", linestyle="--", label="Mean Load")
    plt.title("Monte Carlo Simulation: Material Strength vs Load")
    plt.xlabel("Load / Strength (kg)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation for mechanical load reliability.",
    )
    parser.add_argument(
        "--material-strength-mean",
        type=float,
        default=100.0,
        help="Average material strength (kg).",
    )
    parser.add_argument(
        "--material-strength-std",
        type=float,
        default=10.0,
        help="Standard deviation of material strength.",
    )
    parser.add_argument(
        "--load-mean",
        type=float,
        default=90.0,
        help="Average applied load (kg).",
    )
    parser.add_argument(
        "--load-std",
        type=float,
        default=15.0,
        help="Standard deviation of applied load.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Set for reproducible runs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and print text output only.",
    )
    return parser


def parse_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        material_strength_mean=args.material_strength_mean,
        material_strength_std=args.material_strength_std,
        load_mean=args.load_mean,
        load_std=args.load_std,
        num_simulations=args.num_simulations,
        seed=args.seed,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = parse_config(args)

    try:
        result = run_simulation(config)
    except ValueError as exc:
        parser.error(str(exc))

    print_summary(result)

    if not args.no_plot:
        try:
            plot_simulation(result, config)
        except ModuleNotFoundError as exc:
            if exc.name == "matplotlib":
                print("matplotlib is required for plotting. Install it or pass --no-plot.")
                return 1
            raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
