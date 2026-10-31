from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import beta as  b_dist
from scipy.stats import poisson as p_dist

z90 = 1.281551565544600
LEC_points = 200

@dataclass(frozen=True)
class triple:
    low: float
    mode: float
    high: float
    
    def __post_init__(self):
        if not (self.low <= self.mode <= self.high):
            raise ValueError(f"Invalid triple: harus low <= mode <= high, dapatkan {self}")
        
    def scaled(self, k:float) -> "triple":
        return triple(self.low * k, self.mode * k, self.high * k)
    
@dataclass(frozen=True)
class LogNormalSpec:
    median: float
    p90: float
    
    def __post_init__(self):
        if self.median <= 0 or self.p90 < self.median:
            raise ValueError(f"Perlu 0 < median <= p90, dapat {self}")
    
    @property
    def mu(self) -> float:
        return np.log(self.median)
    
    @property
    def sigma(self) -> float:
        return max(float(np.log(self.p90 / self.median)) / z90, 1e-9)
    
    def scaled(self, k: float) -> "LogNormalSpec":
        return LogNormalSpec(self.median * k, self.p90 * k)

@dataclass(frozen=True)
class fair_input:
    tef: triple
    vulnerability: triple
    primary_loss: LogNormalSpec
    secondary_loss: LogNormalSpec
    secondary_probability: triple
    
    def __post_init__(self):
        if not (0 <= self.vulnerability.low and self.vulnerability.high <= 1):
            raise ValueError(f"Invalid vulnerability: harus di rentan 0 - 1")
        if not (0 <= self.secondary_probability <= 1):
            raise ValueError(f"Invalid secondary_probability: harus di rentan 0 - 1")
        
@dataclass
class hasil_simulasi:
    iteration: int
    seed: int | None
    annual_loss: np.ndarray
    sle: float
    ale: float
    p50: float
    p90: float
    p95: float
    mean_lef: float
    loss_exceedance: dict
    
    def summary(self) -> dict:
        return {"SLE": self.sle, "ALE": self.ale, "P50": self.p50, 
                "P90": self.p90, "P95": self.p95, "Mean LEF": self.mean_lef}
        
def sample_pert(u: np.ndarray, t: triple) -> np.ndarray:
    if t.high == t.low:
        return np.full(u.shape, float(t.low))
    span = t.high - t.low
    a= 1 + 4 * (t.mode - t.low) / span
    b= 1 + 4 * (t.high - t.mode) / span
    return t.low + span * b_dist.ppf(u, a, b)

def sample_lognormal(u: np.ndarray, s: LogNormalSpec) -> np.ndarray:
    return np.exp(s.mu + s.sigma * ndtri(u))

def loss_exceedance_curve(annual: np.ndarray) -> dict:
    xs= np.sort(annual)
    n= len(xs)
    idx= np.unique(np.linspace(0, n-1, min(LEC_points, n)).astype(int))
    return {"Loss": xs[idx], "Probability": (1 - idx / n).tolist()}

def run_simulation(inputs, iteration: int = 10_000, seed: int | None = 42) -> hasil_simulasi:
    n = int(iteration)
    ss = np.random.SeedSequence(seed)
    r_tef, r_vul, r_cnt, r_prim, r_occ, r_sec = (np.random.default_rng(s) for s in ss.spawn(6))
    
    tef = sample_pert(r_tef.random(n), inputs.tef)
    vul = np.clip(sample_pert(r_vul.random(n), inputs.vulnerability), 0, 1)
    lef = tef * vul
    count = p_dist.ppf(np.clip(r_cnt.random(n), 0, 1 - 1e-12), lef).astype(int)
    
    total = int(count.sum())
    primary = sample_lognormal(r_prim.random(total), inputs.primary_loss)
    occurs = r_occ.random(total) < inputs.secondary_probability
    secondary = sample_lognormal(r_sec.random(total), inputs.secondary_loss) * occurs
    event_loss = primary + secondary
    
    owner = np.repeat(np.arange(n), count)
    annual = np.bincount(owner, weights = event_loss, minlength = n)
    
    return hasil_simulasi(
    iteration = n,
    seed = seed,
    annual_loss = annual,
    sle = float(event_loss.mean()) if total else 0.0,
    ale = float(annual.mean()),
    p50 = float(np.percentile(annual, 50)),
    p90 = float(np.percentile(annual, 90)),
    p95 = float(np.percentile(annual, 95)),
    mean_lef = float(lef.mean()),
    loss_exceedance = loss_exceedance_curve(annual)
)
    