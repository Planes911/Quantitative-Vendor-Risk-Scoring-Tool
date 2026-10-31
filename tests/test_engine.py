import pytest
from src.engine import triple, LogNormalSpec, fair_input, run_simulation

def test_triple_valid():
    with pytest.raises(ValueError):
        triple(low= 10, mode= 5, high= 1)
        
def test_lognormal_negatif_median():
    with pytest.raises(ValueError):
        LogNormalSpec(median=-5, p90=-10)
        
def test_hasil_simulasi():
    inp= fair_input(
        tef= triple(0.5, 2, 6),
        vulnerability= triple(0.1, 0.3, 0.6),
        primary_loss= LogNormalSpec(50-000, 200_000),
        secondary_loss= LogNormalSpec(100_000, 500_000),
        secondary_probability= 0.3,
    )
    
    assert run_simulation(inp, 5000, seed=42). ale== run_simulation(inp, 5000, seed=42).ale