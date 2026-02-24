import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (
    Component,
    SimulationConfig,
    System,
    SystemConfig,
    run_simulation,
    run_system_simulation,
)


def test_reproducible_results_with_same_seed():
    config = SimulationConfig(num_simulations=5_000, seed=123)

    first_run = run_simulation(config)
    second_run = run_simulation(config)

    assert np.array_equal(first_run.failures, second_run.failures)
    assert first_run.failure_rate == second_run.failure_rate


def test_failure_rate_moves_with_load_level():
    safer_config = SimulationConfig(
        material_strength_mean=100,
        material_strength_std=1,
        load_mean=80,
        load_std=1,
        num_simulations=5_000,
        seed=7,
    )
    riskier_config = SimulationConfig(
        material_strength_mean=100,
        material_strength_std=1,
        load_mean=120,
        load_std=1,
        num_simulations=5_000,
        seed=7,
    )

    safer_result = run_simulation(safer_config)
    riskier_result = run_simulation(riskier_config)

    assert safer_result.failure_rate < 0.01
    assert riskier_result.failure_rate > 0.99


def test_invalid_configuration_raises_value_error():
    with pytest.raises(ValueError):
        run_simulation(SimulationConfig(material_strength_std=0))

    with pytest.raises(ValueError):
        run_simulation(SimulationConfig(load_std=0))

    with pytest.raises(ValueError):
        run_simulation(SimulationConfig(num_simulations=0))

    with pytest.raises(ValueError):
        run_simulation(SimulationConfig(strength_dist="lognormal", material_strength_mean=0))


def test_confidence_interval_matches_binomial_formula():
    config = SimulationConfig(
        material_strength_mean=100,
        material_strength_std=10,
        load_mean=95,
        load_std=12,
        num_simulations=20_000,
        seed=11,
    )
    result = run_simulation(config)

    p = result.failure_rate
    n = result.failures.size
    half_width = 1.96 * math.sqrt(p * (1.0 - p) / n)
    expected_low = max(0.0, p - half_width)
    expected_high = min(1.0, p + half_width)

    assert math.isclose(result.confidence_interval[0], expected_low, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(result.confidence_interval[1], expected_high, rel_tol=0, abs_tol=1e-12)


def test_closed_form_beta_for_normal_inputs():
    config = SimulationConfig(
        material_strength_mean=120,
        material_strength_std=12,
        load_mean=100,
        load_std=8,
        num_simulations=5_000,
        seed=4,
    )
    result = run_simulation(config)
    expected_beta = (config.material_strength_mean - config.load_mean) / math.sqrt(
        config.material_strength_std**2 + config.load_std**2
    )
    assert math.isclose(result.reliability_index_beta or 0.0, expected_beta, rel_tol=0, abs_tol=1e-12)


def test_non_normal_beta_not_available():
    config = SimulationConfig(
        material_strength_mean=120,
        material_strength_std=12,
        load_mean=100,
        load_std=8,
        strength_dist="weibull",
        load_dist="normal",
        num_simulations=5_000,
        seed=4,
    )
    result = run_simulation(config)
    assert result.reliability_index_beta is None


def test_positive_support_distributions_produce_positive_samples():
    config = SimulationConfig(
        material_strength_mean=120,
        material_strength_std=20,
        load_mean=90,
        load_std=15,
        strength_dist="weibull",
        load_dist="lognormal",
        num_simulations=10_000,
        seed=19,
    )
    result = run_simulation(config)
    assert np.all(result.material_strengths > 0)
    assert np.all(result.loads > 0)


def test_lhs_reproducible_with_same_seed():
    config = SimulationConfig(
        material_strength_mean=100,
        material_strength_std=10,
        load_mean=92,
        load_std=9,
        sampling_method="lhs",
        num_simulations=4_000,
        seed=21,
    )
    first = run_simulation(config)
    second = run_simulation(config)
    assert np.array_equal(first.failures, second.failures)
    assert np.array_equal(first.material_strengths, second.material_strengths)
    assert np.array_equal(first.loads, second.loads)


def test_system_series_parallel_and_kofn_relationship():
    components = (
        Component("Shaft", strength_mean=100, strength_std=10, load_mean=95, load_std=12, dist="normal"),
        Component("Bearing", strength_mean=110, strength_std=9, load_mean=100, load_std=11, dist="normal"),
        Component("Bolt", strength_mean=90, strength_std=8, load_mean=85, load_std=10, dist="normal"),
    )

    series_result = run_system_simulation(
        SystemConfig(
            components=components,
            configuration="series",
            num_simulations=30_000,
            seed=8,
        ),
        include_sensitivity=False,
    )
    parallel_result = run_system_simulation(
        SystemConfig(
            components=components,
            configuration="parallel",
            num_simulations=30_000,
            seed=8,
        ),
        include_sensitivity=False,
    )
    k2_result = run_system_simulation(
        SystemConfig(
            components=components,
            configuration="k-of-n",
            k=2,
            num_simulations=30_000,
            seed=8,
        ),
        include_sensitivity=False,
    )

    assert series_result.failure_rate >= k2_result.failure_rate >= parallel_result.failure_rate
    assert series_result.critical_component in {component.name for component in components}


def test_system_sensitivity_reports_failure_reduction():
    components = (
        Component("A", strength_mean=100, strength_std=10, load_mean=95, load_std=12, dist="normal"),
        Component("B", strength_mean=120, strength_std=10, load_mean=110, load_std=11, dist="normal"),
    )
    result = run_system_simulation(
        SystemConfig(
            components=components,
            configuration="series",
            num_simulations=25_000,
            seed=12,
            strength_uplift=0.10,
        ),
        include_sensitivity=True,
    )

    assert set(result.sensitivity_delta) == {"A", "B"}
    assert result.sensitivity_delta["A"] <= 0
    assert result.sensitivity_delta["B"] <= 0


def test_system_wrapper_api_supports_k_of_n():
    components = (
        Component("A", strength_mean=100, strength_std=10, load_mean=96, load_std=11, dist="normal"),
        Component("B", strength_mean=105, strength_std=10, load_mean=100, load_std=11, dist="normal"),
        Component("C", strength_mean=110, strength_std=10, load_mean=101, load_std=11, dist="normal"),
    )
    system = System(components=components, config="series")
    series_result = system.simulate(n=20_000, seed=22, include_sensitivity=False)

    k_of_n_system = System(components=components, k_of_n=(2, 3))
    k_of_n_result = k_of_n_system.simulate(n=20_000, seed=22, include_sensitivity=False)

    assert series_result.failure_rate >= k_of_n_result.failure_rate
