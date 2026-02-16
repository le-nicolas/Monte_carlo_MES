import numpy as np
import pytest

from main import SimulationConfig, run_simulation


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
