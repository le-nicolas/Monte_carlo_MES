# Monte Carlo MES
Monte Carlo reliability simulation for mechanical systems.

This project estimates failure probability when applied loads can exceed material strength under uncertain conditions. It now supports single-component and multi-component system reliability workflows.

![Mechanical Reliability Monte Carlo - What It Can Do](what-it-can-do.svg)

## Features
- Distribution selection per variable: `normal`, `lognormal`, `weibull`
- Sampling methods: standard Monte Carlo (`mc`) and Latin Hypercube Sampling (`lhs`)
- 95% confidence interval on failure probability using binomial standard error
- Running convergence diagnostics for failure probability
- Safety factor and closed-form normal reliability index `beta`
- Multi-component system reliability:
  - `series` (fails if any component fails)
  - `parallel` (fails if all components fail)
  - `k-of-n` (fails if at least `k` components fail)
- Critical component importance and strength-uplift sensitivity analysis
- Publication-oriented plots for:
  - load vs strength distributions
  - explicit failure region (`load > strength`) via margin plot
  - convergence of failure estimate

## Quick Start
```powershell
git clone https://github.com/le-nicolas/Monte_carlo_MES.git
cd Monte_carlo_MES
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --no-plot
```

## Single-Component CLI
```powershell
python main.py [options]
```

Core options:
- `--material-strength-mean` (default: `100`)
- `--material-strength-std` (default: `10`)
- `--load-mean` (default: `90`)
- `--load-std` (default: `15`)
- `--strength-dist` (`normal|lognormal|weibull`, default: `normal`)
- `--load-dist` (`normal|lognormal|weibull`, default: `normal`)
- `--sampling-method` (`mc|lhs`, default: `mc`)
- `--num-simulations` (default: `10000`)
- `--seed` (default: `42`)
- `--convergence-tol` (default: `0.001`)
- `--min-convergence-samples` (default: `1000`)
- `--min-convergence-events` (default: `5`)
- `--no-plot`

Example:
```powershell
python main.py --strength-dist lognormal --load-dist weibull --sampling-method lhs --num-simulations 50000 --seed 7
```

## Multi-Component System Mode
Pass `--components-file` to switch into system reliability mode.

```powershell
python main.py --components-file components.json --system-config series --num-simulations 100000 --no-plot
```

Supported system options:
- `--components-file <path>`
- `--system-config series|parallel|k-of-n`
- `--k` (required for `k-of-n`)
- `--strength-uplift` (default `0.10`, sensitivity analysis percentage as fraction)
- `--skip-sensitivity`

Programmatic API example:
```python
from main import Component, System

components = [
    Component("Shaft", strength_mean=200, strength_std=20, load_mean=150, load_std=25, dist="normal"),
    Component("Bearing", strength_mean=180, strength_std=15, load_mean=160, load_std=20, dist="lognormal"),
    Component("Bolt", strength_mean=120, strength_std=10, load_mean=100, load_std=18, dist="normal"),
]

system = System(components, config="series")
result = system.simulate(n=100_000)
print(result.failure_rate, result.component_failure_rates, result.critical_component)
```

Example `components.json`:
```json
{
  "components": [
    {
      "name": "Shaft",
      "strength_mean": 200,
      "strength_std": 20,
      "load_mean": 150,
      "load_std": 25,
      "dist": "normal"
    },
    {
      "name": "Bearing",
      "strength_mean": 180,
      "strength_std": 15,
      "load_mean": 160,
      "load_std": 20,
      "dist": "lognormal"
    },
    {
      "name": "Bolt",
      "strength_mean": 120,
      "strength_std": 10,
      "load_mean": 100,
      "load_std": 18,
      "dist": "normal"
    }
  ],
  "system": {
    "config": "k-of-n",
    "k": 2
  }
}
```

## Output
Single-component mode reports:
- failure rate
- 95% confidence interval
- failure count
- reliability
- safety factor
- closed-form reliability index `beta` (normal-only)
- convergence status

System mode reports:
- per-component failure rates
- system failure rate and reliability
- 95% confidence interval
- critical component (importance)
- sensitivity deltas for strength uplift

## Development
Install test dependencies and run tests:
```powershell
pip install -r requirements-dev.txt
pytest
```
