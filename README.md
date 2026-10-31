# Quantitative Vendor Risk Scoring Tool 

A Python tool that estimated financial loss from thrid-party vendor data breaches using FAIR framework and Monte Carlo simulation.

> **Status**: Core simulation engine and unit test are done, Streamlit interface and loss exceedance visualization are planed.

## What the tool does

Instated of a single average loss, the engine simulate thousants of possible years of data breaches events and reports a probability distribution of annual loss.

**Inputs**
- Threat Event Frequency (TEF) and Vulenerability as (low, mode, high) estimates, modeled with PERT distributions
- Primary and secondary loss as log-normal distributions, defined by median and P90
- Probability that secondary loss occurs

**Outputs**
- SLE (Single Loss Expectancy)
- ALE (Annual Loss Expectancy)
- P50 / P90 / P95 of annual loss
- Mean Loss Event Frequency (LEF)
- Loss exceedance curve data

## How the tool works
1. Sample TEF and Vulnerability (PERT), then compute LEF = TEF x Vulnerability
2. Sample the number of loss events per year (Poisson, based on LEF)
3. Sample primary loss, and secondary loss when it occurs, for each event (log-normal)
4. Aggregate loss per simulated year and compute the statistics above

Calculations are vectorized with NumPy/SciPy, and results are reproducible with a fixed seed.

## Installation
```bash
git clone https://github.com/Planes911/Quantitative-Vendor-Risk-Scoring-Tool.git
cd Quantitative-Vendor-Risk-Scoring-Tool
pip install -r requirements.txt
```

## Usage
```python
from src.engine import triple, LogNormalSpec, fair_input, run_simulation

inputs = fair_input(
    tef=triple(0.5, 2, 6),
    vulnerability=triple(0.1, 0.3, 0.6),
    primary_loss=LogNormalSpec(median=50_000, p90=200_000),
    secondary_loss=LogNormalSpec(median=100_000, p90=500_000),
    secondary_probability=0.3,
)

result = run_simulation(inputs, iteration=10_000, seed=42)
print(result.summary())
```

Example output (illustrative values):
```
SLE ~ 150,498 | ALE ~ 115,161 | P50 ~ 2,426 | P90 ~ 340,938 | P95 ~ 533,392 | Mean LEF ~ 0.77
```

## Running tests
```bash
pytest
```
Tests cover input validation and reproducibility.

## Project structure
```
src/engine.py        # FAIR Monte Carlo engine
tests/test_engine.py # unit tests
```

## Notes
Results depend on the input estimates, which are assumptions. The example values are illustrative, not real vendor data.