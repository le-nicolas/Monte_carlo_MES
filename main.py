"""Monte Carlo simulation for mechanical load reliability."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Optional

import numpy as np

Z_95 = 1.96
SUPPORTED_DISTS = ("normal", "lognormal", "weibull")
SUPPORTED_SAMPLING_METHODS = ("mc", "lhs")
SUPPORTED_SYSTEM_CONFIGS = ("series", "parallel", "k-of-n")
_STANDARD_NORMAL = NormalDist()


def _normalize_distribution_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in {"gaussian"}:
        return "normal"
    return normalized


def _normalize_sampling_method(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"latin-hypercube", "latin", "latin-hs"}:
        return "lhs"
    return normalized


def _normalize_system_configuration(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    if normalized in {"kofn", "k-out-of-n", "kofn-system"}:
        return "k-of-n"
    return normalized


def _lognormal_params_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    variance = std**2
    sigma_sq = math.log1p(variance / (mean**2))
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - 0.5 * sigma_sq
    return mu, sigma


def _weibull_cv(shape: float) -> float:
    lg1 = math.lgamma(1.0 + 1.0 / shape)
    lg2 = math.lgamma(1.0 + 2.0 / shape)
    ratio = math.exp(lg2 - 2.0 * lg1)
    return math.sqrt(max(ratio - 1.0, 0.0))


def _weibull_shape_from_mean_std(mean: float, std: float) -> float:
    target_cv = std / mean
    low = 0.2
    high = 10.0

    while _weibull_cv(low) < target_cv:
        low /= 2.0
        if low <= 1e-6:
            break

    while _weibull_cv(high) > target_cv:
        high *= 2.0
        if high >= 1e6:
            break

    for _ in range(120):
        mid = 0.5 * (low + high)
        cv_mid = _weibull_cv(mid)
        if cv_mid > target_cv:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _weibull_params_from_mean_std(mean: float, std: float) -> tuple[float, float]:
    shape = _weibull_shape_from_mean_std(mean, std)
    scale = mean / math.exp(math.lgamma(1.0 + 1.0 / shape))
    return shape, scale


def _standard_normal_ppf(uniform_samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(uniform_samples, 1e-12, 1.0 - 1e-12)
    flat = clipped.ravel()
    values = np.fromiter(
        (_STANDARD_NORMAL.inv_cdf(float(probability)) for probability in flat),
        dtype=float,
        count=flat.size,
    )
    return values.reshape(clipped.shape)


def _ppf_from_distribution(
    uniform_samples: np.ndarray,
    distribution: str,
    mean: float,
    std: float,
) -> np.ndarray:
    distribution_name = _normalize_distribution_name(distribution)

    if distribution_name == "normal":
        return mean + std * _standard_normal_ppf(uniform_samples)

    if distribution_name == "lognormal":
        mu, sigma = _lognormal_params_from_mean_std(mean, std)
        return np.exp(mu + sigma * _standard_normal_ppf(uniform_samples))

    if distribution_name == "weibull":
        shape, scale = _weibull_params_from_mean_std(mean, std)
        clipped = np.clip(uniform_samples, 1e-12, 1.0 - 1e-12)
        return scale * np.power(-np.log1p(-clipped), 1.0 / shape)

    raise ValueError(f"Unsupported distribution: {distribution}")


def _sample_from_distribution(
    rng: np.random.Generator,
    distribution: str,
    mean: float,
    std: float,
    size: int,
) -> np.ndarray:
    distribution_name = _normalize_distribution_name(distribution)

    if distribution_name == "normal":
        return rng.normal(loc=mean, scale=std, size=size)

    if distribution_name == "lognormal":
        mu, sigma = _lognormal_params_from_mean_std(mean, std)
        return rng.lognormal(mean=mu, sigma=sigma, size=size)

    if distribution_name == "weibull":
        shape, scale = _weibull_params_from_mean_std(mean, std)
        return scale * rng.weibull(shape, size=size)

    raise ValueError(f"Unsupported distribution: {distribution}")


def _latin_hypercube(
    rng: np.random.Generator,
    num_samples: int,
    num_dimensions: int,
) -> np.ndarray:
    lhs_samples = np.empty((num_samples, num_dimensions), dtype=float)
    for dimension in range(num_dimensions):
        stratified = (np.arange(num_samples, dtype=float) + rng.random(num_samples)) / num_samples
        rng.shuffle(stratified)
        lhs_samples[:, dimension] = stratified
    return lhs_samples


def _binomial_confidence_interval(
    probability: float,
    sample_count: int,
    z_score: float = Z_95,
) -> tuple[float, float]:
    half_width = z_score * math.sqrt(probability * (1.0 - probability) / sample_count)
    lower = max(0.0, probability - half_width)
    upper = min(1.0, probability + half_width)
    return lower, upper


def _running_failure_statistics(
    failures: np.ndarray,
    convergence_tol: float,
    min_samples: int,
    min_events: int,
) -> tuple[np.ndarray, np.ndarray, Optional[int]]:
    sample_count = failures.size
    trial_index = np.arange(1, sample_count + 1, dtype=float)
    cumulative_failures = np.cumsum(failures.astype(float))
    running_failure_rate = cumulative_failures / trial_index
    running_half_width = Z_95 * np.sqrt(
        np.maximum(running_failure_rate * (1.0 - running_failure_rate), 0.0) / trial_index
    )
    cumulative_non_failures = trial_index - cumulative_failures
    eligible = (
        (trial_index >= min_samples)
        & (cumulative_failures >= min_events)
        & (cumulative_non_failures >= min_events)
        & (running_half_width <= convergence_tol)
    )
    convergence_sample = int(np.argmax(eligible) + 1) if np.any(eligible) else None
    return running_failure_rate, running_half_width, convergence_sample


@dataclass(frozen=True)
class SimulationConfig:
    """Input parameters for single-component reliability analysis."""

    material_strength_mean: float = 100.0
    material_strength_std: float = 10.0
    load_mean: float = 90.0
    load_std: float = 15.0
    num_simulations: int = 10_000
    seed: Optional[int] = 42
    strength_dist: str = "normal"
    load_dist: str = "normal"
    sampling_method: str = "mc"
    convergence_tol: float = 0.001
    min_convergence_samples: int = 1_000
    min_convergence_events: int = 5

    def validate(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if self.material_strength_std <= 0:
            raise ValueError("material_strength_std must be > 0")
        if self.load_std <= 0:
            raise ValueError("load_std must be > 0")
        if self.convergence_tol <= 0:
            raise ValueError("convergence_tol must be > 0")
        if self.min_convergence_samples <= 0:
            raise ValueError("min_convergence_samples must be > 0")
        if self.min_convergence_events <= 0:
            raise ValueError("min_convergence_events must be > 0")

        strength_dist = _normalize_distribution_name(self.strength_dist)
        load_dist = _normalize_distribution_name(self.load_dist)
        if strength_dist not in SUPPORTED_DISTS:
            raise ValueError(f"Unsupported strength_dist: {self.strength_dist}")
        if load_dist not in SUPPORTED_DISTS:
            raise ValueError(f"Unsupported load_dist: {self.load_dist}")

        sampling_method = _normalize_sampling_method(self.sampling_method)
        if sampling_method not in SUPPORTED_SAMPLING_METHODS:
            raise ValueError(f"Unsupported sampling_method: {self.sampling_method}")

        if strength_dist in {"lognormal", "weibull"} and self.material_strength_mean <= 0:
            raise ValueError("material_strength_mean must be > 0 for lognormal/weibull distributions")
        if load_dist in {"lognormal", "weibull"} and self.load_mean <= 0:
            raise ValueError("load_mean must be > 0 for lognormal/weibull distributions")


@dataclass(frozen=True)
class SimulationResult:
    """Computed arrays and aggregate statistics for a single component."""

    material_strengths: np.ndarray
    loads: np.ndarray
    failures: np.ndarray
    confidence_interval: tuple[float, float]
    running_failure_rate: np.ndarray
    running_ci_half_width: np.ndarray
    convergence_sample: Optional[int]
    safety_factor: float
    reliability_index_beta: Optional[float]
    sampling_method: str
    strength_dist: str
    load_dist: str
    convergence_tol: float

    @property
    def failure_rate(self) -> float:
        return float(np.mean(self.failures))

    @property
    def failure_count(self) -> int:
        return int(np.sum(self.failures))

    @property
    def reliability(self) -> float:
        return 1.0 - self.failure_rate


def _closed_form_reliability_index(config: SimulationConfig) -> Optional[float]:
    strength_dist = _normalize_distribution_name(config.strength_dist)
    load_dist = _normalize_distribution_name(config.load_dist)
    if strength_dist != "normal" or load_dist != "normal":
        return None
    denominator = math.sqrt(config.material_strength_std**2 + config.load_std**2)
    return (config.material_strength_mean - config.load_mean) / denominator


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run Monte Carlo simulation and return full output arrays."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    strength_dist = _normalize_distribution_name(config.strength_dist)
    load_dist = _normalize_distribution_name(config.load_dist)
    sampling_method = _normalize_sampling_method(config.sampling_method)

    if sampling_method == "mc":
        material_strengths = _sample_from_distribution(
            rng=rng,
            distribution=strength_dist,
            mean=config.material_strength_mean,
            std=config.material_strength_std,
            size=config.num_simulations,
        )
        loads = _sample_from_distribution(
            rng=rng,
            distribution=load_dist,
            mean=config.load_mean,
            std=config.load_std,
            size=config.num_simulations,
        )
    else:
        lhs_samples = _latin_hypercube(rng, config.num_simulations, 2)
        material_strengths = _ppf_from_distribution(
            uniform_samples=lhs_samples[:, 0],
            distribution=strength_dist,
            mean=config.material_strength_mean,
            std=config.material_strength_std,
        )
        loads = _ppf_from_distribution(
            uniform_samples=lhs_samples[:, 1],
            distribution=load_dist,
            mean=config.load_mean,
            std=config.load_std,
        )

    failures = loads > material_strengths
    failure_rate = float(np.mean(failures))
    confidence_interval = _binomial_confidence_interval(failure_rate, failures.size)
    running_failure_rate, running_ci_half_width, convergence_sample = _running_failure_statistics(
        failures=failures,
        convergence_tol=config.convergence_tol,
        min_samples=config.min_convergence_samples,
        min_events=config.min_convergence_events,
    )
    safety_factor = math.inf if config.load_mean == 0 else config.material_strength_mean / config.load_mean

    return SimulationResult(
        material_strengths=material_strengths,
        loads=loads,
        failures=failures,
        confidence_interval=confidence_interval,
        running_failure_rate=running_failure_rate,
        running_ci_half_width=running_ci_half_width,
        convergence_sample=convergence_sample,
        safety_factor=safety_factor,
        reliability_index_beta=_closed_form_reliability_index(config),
        sampling_method=sampling_method,
        strength_dist=strength_dist,
        load_dist=load_dist,
        convergence_tol=config.convergence_tol,
    )

@dataclass(frozen=True)
class Component:
    """Component-level stochastic strength and load definition."""

    name: str
    strength_mean: float
    strength_std: float
    load_mean: float
    load_std: float
    dist: Optional[str] = None
    strength_dist: Optional[str] = None
    load_dist: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "Component":
        def pick_required(*keys: str) -> float:
            for key in keys:
                if key in payload:
                    return float(payload[key])
            keys_text = ", ".join(keys)
            raise ValueError(f"Component is missing required key(s): {keys_text}")

        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Each component must define a non-empty 'name'")

        return cls(
            name=name,
            strength_mean=pick_required("strength_mean", "material_strength_mean"),
            strength_std=pick_required("strength_std", "material_strength_std"),
            load_mean=pick_required("load_mean"),
            load_std=pick_required("load_std"),
            dist=payload.get("dist"),
            strength_dist=payload.get("strength_dist"),
            load_dist=payload.get("load_dist"),
        )

    @property
    def resolved_strength_dist(self) -> str:
        selected = self.strength_dist if self.strength_dist is not None else self.dist or "normal"
        return _normalize_distribution_name(selected)

    @property
    def resolved_load_dist(self) -> str:
        selected = self.load_dist if self.load_dist is not None else self.dist or "normal"
        return _normalize_distribution_name(selected)

    def validate(self) -> None:
        if self.strength_std <= 0:
            raise ValueError(f"{self.name}: strength_std must be > 0")
        if self.load_std <= 0:
            raise ValueError(f"{self.name}: load_std must be > 0")

        strength_dist = self.resolved_strength_dist
        load_dist = self.resolved_load_dist
        if strength_dist not in SUPPORTED_DISTS:
            raise ValueError(f"{self.name}: unsupported strength distribution '{strength_dist}'")
        if load_dist not in SUPPORTED_DISTS:
            raise ValueError(f"{self.name}: unsupported load distribution '{load_dist}'")

        if strength_dist in {"lognormal", "weibull"} and self.strength_mean <= 0:
            raise ValueError(f"{self.name}: strength_mean must be > 0 for lognormal/weibull")
        if load_dist in {"lognormal", "weibull"} and self.load_mean <= 0:
            raise ValueError(f"{self.name}: load_mean must be > 0 for lognormal/weibull")


@dataclass(frozen=True)
class SystemConfig:
    """Input parameters for multi-component system reliability analysis."""

    components: tuple[Component, ...]
    configuration: str = "series"
    k: Optional[int] = None
    num_simulations: int = 10_000
    seed: Optional[int] = 42
    sampling_method: str = "mc"
    convergence_tol: float = 0.001
    min_convergence_samples: int = 1_000
    min_convergence_events: int = 5
    strength_uplift: float = 0.10

    def validate(self) -> None:
        if not self.components:
            raise ValueError("At least one component is required")
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")
        if self.convergence_tol <= 0:
            raise ValueError("convergence_tol must be > 0")
        if self.min_convergence_samples <= 0:
            raise ValueError("min_convergence_samples must be > 0")
        if self.min_convergence_events <= 0:
            raise ValueError("min_convergence_events must be > 0")
        if self.strength_uplift <= -1:
            raise ValueError("strength_uplift must be > -1")

        sampling_method = _normalize_sampling_method(self.sampling_method)
        if sampling_method not in SUPPORTED_SAMPLING_METHODS:
            raise ValueError(f"Unsupported sampling_method: {self.sampling_method}")

        normalized_configuration = _normalize_system_configuration(self.configuration)
        if normalized_configuration not in SUPPORTED_SYSTEM_CONFIGS:
            raise ValueError(f"Unsupported configuration: {self.configuration}")

        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Component names must be unique")

        for component in self.components:
            component.validate()

        if normalized_configuration == "k-of-n":
            if self.k is None:
                raise ValueError("k must be set for k-of-n systems")
            if self.k <= 0:
                raise ValueError("k must be > 0 for k-of-n systems")
            if self.k > len(self.components):
                raise ValueError("k cannot exceed number of components")


class System:
    """Convenience wrapper matching engineering workflow style usage."""

    def __init__(
        self,
        components: list[Component] | tuple[Component, ...],
        config: str = "series",
        k_of_n: Optional[tuple[int, int]] = None,
    ) -> None:
        self.components = tuple(components)
        self.config = config
        self.k_of_n = k_of_n

    def simulate(
        self,
        n: int = 10_000,
        seed: Optional[int] = 42,
        sampling_method: str = "mc",
        convergence_tol: float = 0.001,
        min_convergence_samples: int = 1_000,
        min_convergence_events: int = 5,
        strength_uplift: float = 0.10,
        include_sensitivity: bool = True,
    ) -> "SystemSimulationResult":
        configuration = self.config
        k_value: Optional[int] = None
        if self.k_of_n is not None:
            k_value, component_count = self.k_of_n
            if component_count != len(self.components):
                raise ValueError("k_of_n component count must match number of provided components")
            configuration = "k-of-n"

        config = SystemConfig(
            components=self.components,
            configuration=configuration,
            k=k_value,
            num_simulations=n,
            seed=seed,
            sampling_method=sampling_method,
            convergence_tol=convergence_tol,
            min_convergence_samples=min_convergence_samples,
            min_convergence_events=min_convergence_events,
            strength_uplift=strength_uplift,
        )
        return run_system_simulation(config, include_sensitivity=include_sensitivity)


@dataclass(frozen=True)
class SystemSimulationResult:
    """Computed arrays and aggregate statistics for a multi-component system."""

    component_failures: dict[str, np.ndarray]
    component_importance: dict[str, float]
    sensitivity_delta: dict[str, float]
    system_failures: np.ndarray
    confidence_interval: tuple[float, float]
    running_failure_rate: np.ndarray
    running_ci_half_width: np.ndarray
    convergence_sample: Optional[int]
    configuration: str
    k: Optional[int]
    sampling_method: str
    convergence_tol: float
    strength_uplift: float

    @property
    def failure_rate(self) -> float:
        return float(np.mean(self.system_failures))

    @property
    def failure_count(self) -> int:
        return int(np.sum(self.system_failures))

    @property
    def reliability(self) -> float:
        return 1.0 - self.failure_rate

    @property
    def component_failure_rates(self) -> dict[str, float]:
        return {name: float(np.mean(failures)) for name, failures in self.component_failures.items()}

    @property
    def critical_component(self) -> Optional[str]:
        if not self.component_importance:
            return None
        return max(self.component_importance, key=self.component_importance.get)


def _evaluate_system_failures(
    component_failure_matrix: np.ndarray,
    configuration: str,
    k: Optional[int],
) -> np.ndarray:
    normalized_configuration = _normalize_system_configuration(configuration)
    if normalized_configuration == "series":
        return np.any(component_failure_matrix, axis=1)
    if normalized_configuration == "parallel":
        return np.all(component_failure_matrix, axis=1)
    if normalized_configuration == "k-of-n":
        threshold = 1 if k is None else k
        return np.sum(component_failure_matrix, axis=1) >= threshold
    raise ValueError(f"Unsupported configuration: {configuration}")


def _simulate_component_failure_matrix(config: SystemConfig) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(config.seed)
    sampling_method = _normalize_sampling_method(config.sampling_method)
    component_count = len(config.components)
    failures = np.zeros((config.num_simulations, component_count), dtype=bool)
    component_names = [component.name for component in config.components]

    lhs_samples = None
    if sampling_method == "lhs":
        lhs_samples = _latin_hypercube(rng, config.num_simulations, 2 * component_count)

    for index, component in enumerate(config.components):
        if sampling_method == "mc":
            strengths = _sample_from_distribution(
                rng=rng,
                distribution=component.resolved_strength_dist,
                mean=component.strength_mean,
                std=component.strength_std,
                size=config.num_simulations,
            )
            loads = _sample_from_distribution(
                rng=rng,
                distribution=component.resolved_load_dist,
                mean=component.load_mean,
                std=component.load_std,
                size=config.num_simulations,
            )
        else:
            assert lhs_samples is not None
            strengths = _ppf_from_distribution(
                uniform_samples=lhs_samples[:, index * 2],
                distribution=component.resolved_strength_dist,
                mean=component.strength_mean,
                std=component.strength_std,
            )
            loads = _ppf_from_distribution(
                uniform_samples=lhs_samples[:, index * 2 + 1],
                distribution=component.resolved_load_dist,
                mean=component.load_mean,
                std=component.load_std,
            )
        failures[:, index] = loads > strengths

    return failures, component_names


def _component_importance(
    component_failure_matrix: np.ndarray,
    component_names: list[str],
    configuration: str,
    k: Optional[int],
) -> dict[str, float]:
    importance: dict[str, float] = {}
    for index, name in enumerate(component_names):
        forced_failures = component_failure_matrix.copy()
        forced_failures[:, index] = True
        pf_if_failed = float(
            np.mean(_evaluate_system_failures(forced_failures, configuration=configuration, k=k))
        )

        forced_safe = component_failure_matrix.copy()
        forced_safe[:, index] = False
        pf_if_safe = float(np.mean(_evaluate_system_failures(forced_safe, configuration=configuration, k=k)))

        importance[name] = pf_if_failed - pf_if_safe
    return importance


def _system_failure_rate_for_config(config: SystemConfig) -> float:
    matrix, _ = _simulate_component_failure_matrix(config)
    failures = _evaluate_system_failures(matrix, configuration=config.configuration, k=config.k)
    return float(np.mean(failures))


def run_system_simulation(
    config: SystemConfig,
    include_sensitivity: bool = True,
) -> SystemSimulationResult:
    """Run Monte Carlo simulation for a multi-component reliability system."""
    config.validate()
    failure_matrix, component_names = _simulate_component_failure_matrix(config)
    system_failures = _evaluate_system_failures(
        component_failure_matrix=failure_matrix,
        configuration=config.configuration,
        k=config.k,
    )

    failure_rate = float(np.mean(system_failures))
    confidence_interval = _binomial_confidence_interval(failure_rate, system_failures.size)
    running_failure_rate, running_ci_half_width, convergence_sample = _running_failure_statistics(
        failures=system_failures,
        convergence_tol=config.convergence_tol,
        min_samples=config.min_convergence_samples,
        min_events=config.min_convergence_events,
    )
    component_importance = _component_importance(
        component_failure_matrix=failure_matrix,
        component_names=component_names,
        configuration=config.configuration,
        k=config.k,
    )

    sensitivity_delta: dict[str, float] = {}
    if include_sensitivity and config.strength_uplift != 0:
        for index, component in enumerate(config.components):
            improved_components = list(config.components)
            improved_components[index] = replace(
                component,
                strength_mean=component.strength_mean * (1.0 + config.strength_uplift),
            )
            improved_config = replace(config, components=tuple(improved_components))
            improved_failure_rate = _system_failure_rate_for_config(improved_config)
            sensitivity_delta[component.name] = improved_failure_rate - failure_rate

    return SystemSimulationResult(
        component_failures={
            component_name: failure_matrix[:, index].copy()
            for index, component_name in enumerate(component_names)
        },
        component_importance=component_importance,
        sensitivity_delta=sensitivity_delta,
        system_failures=system_failures,
        confidence_interval=confidence_interval,
        running_failure_rate=running_failure_rate,
        running_ci_half_width=running_ci_half_width,
        convergence_sample=convergence_sample,
        configuration=_normalize_system_configuration(config.configuration),
        k=config.k,
        sampling_method=_normalize_sampling_method(config.sampling_method),
        convergence_tol=config.convergence_tol,
        strength_uplift=config.strength_uplift,
    )


def _format_percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def print_summary(result: SimulationResult) -> None:
    print(f"Strength Distribution: {result.strength_dist}")
    print(f"Load Distribution: {result.load_dist}")
    print(f"Sampling Method: {result.sampling_method}")
    print(f"Failure Rate: {_format_percent(result.failure_rate)}")
    print(
        "95% CI: "
        f"[{_format_percent(result.confidence_interval[0])}, {_format_percent(result.confidence_interval[1])}]"
    )
    print(f"Failure Count: {result.failure_count}/{result.failures.size}")
    print(f"Reliability: {_format_percent(result.reliability)}")
    print(f"Safety Factor (mu_strength/mu_load): {result.safety_factor:.4f}")
    if result.reliability_index_beta is None:
        print("Closed-Form Reliability Index beta: N/A (requires normal strength/load)")
    else:
        print(f"Closed-Form Reliability Index beta: {result.reliability_index_beta:.4f}")
    if result.convergence_sample is None:
        print(f"Convergence: not reached (target CI half-width <= {result.convergence_tol:.6f})")
    else:
        print(
            "Convergence: "
            f"reached at sample {result.convergence_sample} "
            f"(target CI half-width <= {result.convergence_tol:.6f})"
        )


def print_system_summary(result: SystemSimulationResult) -> None:
    configuration_text = result.configuration
    if result.configuration == "k-of-n":
        component_count = len(result.component_failures)
        configuration_text = f"k-of-n (k={result.k}, n={component_count})"

    print(f"System Configuration: {configuration_text}")
    print(f"Sampling Method: {result.sampling_method}")
    print("Component Failure Rates:")
    for component_name, component_rate in result.component_failure_rates.items():
        print(f"  {component_name}: {_format_percent(component_rate)}")
    print(f"System Failure Rate: {_format_percent(result.failure_rate)}")
    print(f"System Reliability: {_format_percent(result.reliability)}")
    print(
        "95% CI: "
        f"[{_format_percent(result.confidence_interval[0])}, {_format_percent(result.confidence_interval[1])}]"
    )
    if result.convergence_sample is None:
        print(f"Convergence: not reached (target CI half-width <= {result.convergence_tol:.6f})")
    else:
        print(
            "Convergence: "
            f"reached at sample {result.convergence_sample} "
            f"(target CI half-width <= {result.convergence_tol:.6f})"
        )

    if result.critical_component is not None:
        importance = result.component_importance[result.critical_component]
        print(f"Critical Component: {result.critical_component} (importance={importance:.4f})")

    if result.sensitivity_delta:
        uplift_percent = result.strength_uplift * 100
        print(f"Sensitivity ({uplift_percent:.1f}% strength-mean increase):")
        for component_name, delta_failure in result.sensitivity_delta.items():
            print(
                f"  {component_name}: {_format_percent(delta_failure)} "
                "(absolute system failure-rate change)"
            )


def _safe_linear_space(minimum: float, maximum: float, count: int) -> np.ndarray:
    if math.isclose(minimum, maximum):
        return np.linspace(minimum - 0.5, maximum + 0.5, count)
    return np.linspace(minimum, maximum, count)


def plot_simulation(result: SimulationResult, config: SimulationConfig) -> None:
    """Plot distributions, failure region, and convergence diagnostics."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    distribution_ax, margin_ax, convergence_ax = axes

    all_values = np.concatenate([result.material_strengths, result.loads])
    dist_bins = _safe_linear_space(float(np.min(all_values)), float(np.max(all_values)), 45)
    strength_hist, dist_edges = np.histogram(result.material_strengths, bins=dist_bins, density=True)
    load_hist, _ = np.histogram(result.loads, bins=dist_bins, density=True)
    dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])
    overlap = np.minimum(strength_hist, load_hist)

    distribution_ax.plot(dist_centers, strength_hist, color="tab:blue", label="Strength density")
    distribution_ax.plot(dist_centers, load_hist, color="tab:orange", label="Load density")
    distribution_ax.fill_between(
        dist_centers,
        overlap,
        color="tab:red",
        alpha=0.25,
        label="Failure Region (overlap)",
    )
    distribution_ax.axvline(
        config.material_strength_mean,
        color="tab:blue",
        linestyle="--",
        linewidth=1.0,
    )
    distribution_ax.axvline(config.load_mean, color="tab:orange", linestyle="--", linewidth=1.0)
    distribution_ax.set_title("Load vs Strength Densities")
    distribution_ax.set_xlabel("Value")
    distribution_ax.set_ylabel("Density")
    distribution_ax.grid(alpha=0.25)
    distribution_ax.legend()

    margins = result.material_strengths - result.loads
    margin_bins = _safe_linear_space(float(np.min(margins)), float(np.max(margins)), 45)
    margin_hist, margin_edges = np.histogram(margins, bins=margin_bins, density=True)
    margin_centers = 0.5 * (margin_edges[:-1] + margin_edges[1:])
    margin_ax.plot(margin_centers, margin_hist, color="tab:green", label="Margin density")
    margin_ax.fill_between(
        margin_centers,
        0.0,
        margin_hist,
        where=margin_centers < 0,
        color="tab:red",
        alpha=0.35,
        label="Failure Region (Load > Strength)",
    )
    margin_ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    margin_ax.set_title("Reliability Margin")
    margin_ax.set_xlabel("Strength - Load")
    margin_ax.set_ylabel("Density")
    margin_ax.grid(alpha=0.25)
    margin_ax.legend()

    samples = np.arange(1, result.failures.size + 1)
    lower = np.clip(result.running_failure_rate - result.running_ci_half_width, 0.0, 1.0)
    upper = np.clip(result.running_failure_rate + result.running_ci_half_width, 0.0, 1.0)
    convergence_ax.plot(samples, result.running_failure_rate, color="tab:purple", label="Running failure rate")
    convergence_ax.fill_between(samples, lower, upper, color="tab:purple", alpha=0.20, label="95% CI band")
    convergence_ax.axhline(result.failure_rate, color="tab:red", linestyle="--", label="Final failure rate")
    if result.convergence_sample is not None:
        convergence_ax.axvline(
            result.convergence_sample,
            color="tab:green",
            linestyle="--",
            label=f"Converged at {result.convergence_sample}",
        )
    convergence_ax.set_title("Convergence of Failure Probability")
    convergence_ax.set_xlabel("Samples")
    convergence_ax.set_ylabel("Failure Probability")
    convergence_ax.grid(alpha=0.25)
    convergence_ax.legend()

    figure.suptitle("Monte Carlo Mechanical Reliability")
    figure.tight_layout()
    plt.show()


def plot_system_convergence(result: SystemSimulationResult) -> None:
    """Plot running system failure rate and confidence interval."""
    import matplotlib.pyplot as plt

    samples = np.arange(1, result.system_failures.size + 1)
    lower = np.clip(result.running_failure_rate - result.running_ci_half_width, 0.0, 1.0)
    upper = np.clip(result.running_failure_rate + result.running_ci_half_width, 0.0, 1.0)

    plt.figure(figsize=(10, 5))
    plt.plot(samples, result.running_failure_rate, label="Running system failure rate", color="tab:blue")
    plt.fill_between(samples, lower, upper, alpha=0.2, color="tab:blue", label="95% CI band")
    plt.axhline(result.failure_rate, color="tab:red", linestyle="--", label="Final failure rate")
    if result.convergence_sample is not None:
        plt.axvline(result.convergence_sample, color="tab:green", linestyle="--", label="Convergence sample")
    plt.title("System Reliability Convergence")
    plt.xlabel("Samples")
    plt.ylabel("Failure Probability")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _load_components_file(path: str) -> tuple[tuple[Component, ...], Optional[str], Optional[int]]:
    file_path = Path(path)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read components file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in components file: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Components file must contain a top-level JSON object")

    raw_components = payload.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("Components file must include a non-empty 'components' list")

    components = []
    for raw_component in raw_components:
        if not isinstance(raw_component, dict):
            raise ValueError("Each component must be a JSON object")
        components.append(Component.from_mapping(raw_component))

    raw_configuration: Optional[str] = None
    raw_k: Optional[int] = None

    system_block = payload.get("system")
    if isinstance(system_block, dict):
        raw_configuration = system_block.get("config", system_block.get("type"))
        raw_k_value = system_block.get("k")
        if raw_k_value is not None:
            raw_k = int(raw_k_value)

    if raw_configuration is None and "system_config" in payload:
        raw_configuration = str(payload["system_config"])
    if raw_configuration is None and "config" in payload:
        raw_configuration = str(payload["config"])
    if raw_k is None and "k" in payload:
        raw_k = int(payload["k"])

    return tuple(components), raw_configuration, raw_k


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
        "--strength-dist",
        type=str,
        default="normal",
        help="Strength distribution: normal, lognormal, or weibull.",
    )
    parser.add_argument(
        "--load-dist",
        type=str,
        default="normal",
        help="Load distribution: normal, lognormal, or weibull.",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        default="mc",
        help="Sampling approach: mc or lhs.",
    )
    parser.add_argument(
        "--convergence-tol",
        type=float,
        default=0.001,
        help="Convergence target for 95%% CI half-width on running failure rate.",
    )
    parser.add_argument(
        "--min-convergence-samples",
        type=int,
        default=1_000,
        help="Minimum samples before convergence can be reported.",
    )
    parser.add_argument(
        "--min-convergence-events",
        type=int,
        default=5,
        help="Minimum failures and non-failures before convergence can be reported.",
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
        "--components-file",
        type=str,
        default=None,
        help="JSON file for multi-component system reliability analysis.",
    )
    parser.add_argument(
        "--system-config",
        type=str,
        default=None,
        help="System configuration: series, parallel, or k-of-n.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k threshold for k-of-n system configuration.",
    )
    parser.add_argument(
        "--strength-uplift",
        type=float,
        default=0.10,
        help="Fractional strength-mean increase used for sensitivity analysis.",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip component sensitivity analysis in multi-component mode.",
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
        strength_dist=args.strength_dist,
        load_dist=args.load_dist,
        sampling_method=args.sampling_method,
        convergence_tol=args.convergence_tol,
        min_convergence_samples=args.min_convergence_samples,
        min_convergence_events=args.min_convergence_events,
    )


def _build_system_config(args: argparse.Namespace) -> SystemConfig:
    if args.components_file is None:
        raise ValueError("components_file must be provided for system mode")

    components, file_configuration, file_k = _load_components_file(args.components_file)
    system_configuration = args.system_config if args.system_config is not None else file_configuration
    if system_configuration is None:
        system_configuration = "series"
    k_value = args.k if args.k is not None else file_k

    return SystemConfig(
        components=components,
        configuration=system_configuration,
        k=k_value,
        num_simulations=args.num_simulations,
        seed=args.seed,
        sampling_method=args.sampling_method,
        convergence_tol=args.convergence_tol,
        min_convergence_samples=args.min_convergence_samples,
        min_convergence_events=args.min_convergence_events,
        strength_uplift=args.strength_uplift,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.components_file:
        try:
            system_config = _build_system_config(args)
            system_result = run_system_simulation(
                config=system_config,
                include_sensitivity=not args.skip_sensitivity,
            )
        except ValueError as exc:
            parser.error(str(exc))

        print_system_summary(system_result)

        if not args.no_plot:
            try:
                plot_system_convergence(system_result)
            except ModuleNotFoundError as exc:
                if exc.name == "matplotlib":
                    print("matplotlib is required for plotting. Install it or pass --no-plot.")
                    return 1
                raise
        return 0

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
