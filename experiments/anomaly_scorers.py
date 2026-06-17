#!/usr/bin/env python3
"""
Anomaly scorer zoo for CAP-Dedup's Conformal Layer 0.

A scorer is any callable that returns higher values for "more anomalous" samples.
The ConformalAnomalyGate plugs into any scorer and gives the same finite-sample
recall guarantee — but the strength of the guarantee-vs-savings trade-off depends
entirely on how DISCRIMINATIVE the scorer is. A scorer that can't separate faults
from normals will force the gate to preserve nearly everything to meet its target.

Scorers in this module:

    BNNMeanScorer        - mean of Bayesian ensemble predictions (free; uses
                           framework.bayesian_ensemble already trained for L2).
    BNNVarianceScorer    - variance of ensemble predictions (epistemic uncertainty,
                           same metric used for Level 2 gating).
    BNNCombinedScorer    - mean + lambda * variance (combines prediction and
                           uncertainty signals).
    IsolationForestScorer- sklearn IsolationForest (well-known ICS anomaly
                           detector; paper reports ~96% recall as a baseline,
                           so it IS discriminative on TEP).
    AutoencoderScorer    - reconstruction error from a small MLP autoencoder
                           trained on normal samples only (the de-facto ICS
                           heuristic in the literature).

All scorers implement:
    fit(X_train, y_train, framework=None) -> self
    score(X)                              -> 1D float ndarray (higher = more anomalous)

`framework` is passed in for scorers that re-use the existing UncertaintyAwareFramework
artefacts (the BNN scorers). It's ignored by IsolationForest and Autoencoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("anomaly_scorers")


# ---------------------------------------------------------------------------
# 1. BNN-based scorers (reuse trained framework, zero extra training)
# ---------------------------------------------------------------------------

@dataclass
class BNNMeanScorer:
    """Anomaly score = E[BNN_ensemble(x)] = predicted P(fault | x).

    Uses the framework's already-trained Bayesian ensemble. Cheap (only MC-dropout
    cost at scoring time).
    """
    framework: object = None
    mc_samples: Optional[int] = None
    name: str = "bnn_mean"

    def fit(self, X_train=None, y_train=None, framework=None, seed: int = 0):
        if framework is not None:
            self.framework = framework
        if self.framework is None:
            raise ValueError("BNNMeanScorer requires a trained framework")
        if self.mc_samples is None:
            self.mc_samples = self.framework.config["uncertainty"]["mc_samples"]
        return self

    def score(self, X):
        import torch
        fw = self.framework
        fw.bayesian_ensemble.train()  # keep dropout active for MC sampling
        X_t = torch.FloatTensor(np.asarray(X)).to(fw.device)
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(fw.bayesian_ensemble(X_t))
        return torch.cat(preds, dim=0).mean(dim=0).cpu().numpy().flatten()


@dataclass
class BNNVarianceScorer:
    """Anomaly score = Var[BNN_ensemble(x)] = epistemic uncertainty.

    Same signal that Level 2 uses for its gating decision. Tests whether epistemic
    uncertainty alone is a better anomaly proxy than the BNN mean.
    """
    framework: object = None
    mc_samples: Optional[int] = None
    name: str = "bnn_variance"

    def fit(self, X_train=None, y_train=None, framework=None, seed: int = 0):
        if framework is not None:
            self.framework = framework
        if self.framework is None:
            raise ValueError("BNNVarianceScorer requires a trained framework")
        if self.mc_samples is None:
            self.mc_samples = self.framework.config["uncertainty"]["mc_samples"]
        return self

    def score(self, X):
        import torch
        fw = self.framework
        fw.bayesian_ensemble.train()
        X_t = torch.FloatTensor(np.asarray(X)).to(fw.device)
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(fw.bayesian_ensemble(X_t))
        return torch.cat(preds, dim=0).var(dim=0).cpu().numpy().flatten()


@dataclass
class BNNCombinedScorer:
    """Anomaly score = mean(BNN) + lambda * variance(BNN), per-sample normalised.

    The intuition: a fault sample should have either high predicted P(fault)
    or high epistemic uncertainty (or both). The combination is more robust
    than either alone.
    """
    framework: object = None
    mc_samples: Optional[int] = None
    lam: float = 0.5
    name: str = "bnn_combined"

    def fit(self, X_train=None, y_train=None, framework=None, seed: int = 0):
        if framework is not None:
            self.framework = framework
        if self.framework is None:
            raise ValueError("BNNCombinedScorer requires a trained framework")
        if self.mc_samples is None:
            self.mc_samples = self.framework.config["uncertainty"]["mc_samples"]
        return self

    def score(self, X):
        import torch
        fw = self.framework
        fw.bayesian_ensemble.train()
        X_t = torch.FloatTensor(np.asarray(X)).to(fw.device)
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                preds.append(fw.bayesian_ensemble(X_t))
        cat = torch.cat(preds, dim=0)
        m = cat.mean(dim=0).cpu().numpy().flatten()
        v = cat.var(dim=0).cpu().numpy().flatten()
        # Standardise each signal to roughly comparable scales before mixing
        def _z(a):
            mu, sd = float(a.mean()), float(a.std() + 1e-12)
            return (a - mu) / sd
        return _z(m) + self.lam * _z(v)


# ---------------------------------------------------------------------------
# 2. IsolationForest scorer (unsupervised tree ensemble)
# ---------------------------------------------------------------------------

@dataclass
class IsolationForestScorer:
    """Anomaly score = - IsolationForest.score_samples(X).

    sklearn returns higher score_samples for MORE NORMAL points, so we negate
    to make "higher = more anomalous". The paper reports IsolationForest hits
    96% recall on TEP as a baseline, so this scorer IS discriminative on TEP.

    NOTE on reproducibility: random_state is set by the caller via fit(seed=...);
    DO NOT use a fixed default - that would replicate the SAME tree structure
    across every experiment seed and understate the scorer's variance.
    """
    n_estimators: int = 200
    contamination: float = 0.1  # rough fault prevalence in train; tuned on calib
    name: str = "isolation_forest"
    _iso: object = field(default=None, repr=False)
    _seed: int = field(default=42, repr=False)  # set at fit() time

    def fit(self, X_train, y_train=None, framework=None, seed: int = 42):
        from sklearn.ensemble import IsolationForest
        self._seed = int(seed)
        if y_train is not None and (y_train == 0).any():
            X_norm = np.asarray(X_train)[np.asarray(y_train) == 0]
        else:
            X_norm = np.asarray(X_train)
        self._iso = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self._seed,   # <-- per-experiment seed, not fixed
            n_jobs=-1,
        )
        self._iso.fit(X_norm)
        return self

    def score(self, X):
        # IsolationForest.score_samples returns higher = more NORMAL
        # Negate so "higher = more anomalous", consistent with the gate convention.
        return -self._iso.score_samples(np.asarray(X))


# ---------------------------------------------------------------------------
# 3. Autoencoder reconstruction error scorer (de-facto ICS heuristic)
# ---------------------------------------------------------------------------

class AutoencoderScorer:
    """Anomaly score = ||x - autoencoder(x)||_2.

    Trained on normal samples only. A fault sample reconstructs poorly because
    it lies off the learned normal manifold.

    Architecture: input -> 32 -> 16 -> 32 -> input. Small + fast on TEP's 52
    features. Trained for `epochs` (default 50) with AdamW + early stopping.
    """

    def __init__(self, hidden_dim=32, bottleneck_dim=16, epochs=50, batch_size=64,
                 lr=1e-3, weight_decay=1e-5, patience=10, device=None, name="autoencoder"):
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.name = name
        self._model = None
        self._input_dim = None

    def _build(self, input_dim):
        import torch
        import torch.nn as nn
        if self.device is None:
            self.device = torch.device("cpu")  # match framework default
        self._input_dim = input_dim
        self._model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.bottleneck_dim), nn.ReLU(),
            nn.Linear(self.bottleneck_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim),
        ).to(self.device)

    def fit(self, X_train, y_train=None, framework=None, seed: int = 0):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._seed = int(seed)
        # Train on normals only (standard one-class AE recipe)
        X = np.asarray(X_train, dtype=np.float32)
        if y_train is not None and (y_train == 0).any():
            X = X[np.asarray(y_train) == 0]
        if len(X) == 0:
            raise ValueError("AutoencoderScorer.fit: no normal training samples")

        # Seed AE weights init + dropout via torch global state. We accept the
        # outer seed and apply it BEFORE building the model so weight init is
        # reproducible per experiment seed.
        torch.manual_seed(self._seed)
        self._build(X.shape[1])
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X).to(self.device)
        ds = TensorDataset(X_t, X_t)
        # Hold out 10% as internal val for early stopping
        n = len(ds)
        n_val = max(1, n // 10)
        n_tr = n - n_val
        gen = torch.Generator().manual_seed(self._seed)  # per-experiment seed
        tr_ds, val_ds = torch.utils.data.random_split(ds, [n_tr, n_val], generator=gen)
        tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        best_val = float("inf")
        patience_left = self.patience
        for epoch in range(self.epochs):
            self._model.train()
            for xb, _ in tr_loader:
                opt.zero_grad()
                rec = self._model(xb)
                loss = criterion(rec, xb)
                loss.backward()
                opt.step()
            # Validation
            self._model.eval()
            with torch.no_grad():
                vlosses = []
                for xb, _ in val_loader:
                    vlosses.append(criterion(self._model(xb), xb).item())
                vloss = float(np.mean(vlosses)) if vlosses else 0.0
            if vloss + 1e-6 < best_val:
                best_val = vloss
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    logger.info(f"AutoencoderScorer early-stop at epoch {epoch+1} (val={vloss:.5f})")
                    break
        logger.info(f"AutoencoderScorer trained: {epoch+1} epochs, best_val={best_val:.5f}")
        return self

    def score(self, X):
        import torch
        if self._model is None:
            raise RuntimeError("AutoencoderScorer.score called before fit")
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        self._model.eval()
        with torch.no_grad():
            rec = self._model(X_t)
        err = ((rec - X_t) ** 2).mean(dim=1).cpu().numpy().flatten()
        return err  # higher = worse reconstruction = more anomalous


# ---------------------------------------------------------------------------
# 4. ECOD scorer (Empirical CDF Outlier Detection, KDD 2022; SOTA on tabular)
# ---------------------------------------------------------------------------

class ECODScorer:
    """Anomaly score via PyOD's ECOD (Empirical Cumulative Distribution).

    ECOD (Li et al., KDD 2022) is a parameter-free, fast outlier detector that
    estimates per-feature tail probabilities from the empirical CDF. It is the
    current state-of-the-art on tabular anomaly benchmarks (notably ODDS).
    Cost: O(n d log n) fit, O(n d) score; no hyperparameters.
    """

    def __init__(self, name="ecod"):
        self.name = name
        self._model = None

    def fit(self, X_train, y_train=None, framework=None, seed: int = 0):
        # ECOD is unsupervised and DETERMINISTIC (no randomness in the
        # empirical CDF estimation), so the seed is unused. We still accept it
        # so the scorer interface is uniform across all scorers.
        from pyod.models.ecod import ECOD
        X = np.asarray(X_train, dtype=np.float64)
        if y_train is not None and (y_train == 0).any():
            X = X[np.asarray(y_train) == 0]
        self._model = ECOD()
        self._model.fit(X)
        return self

    def score(self, X):
        if self._model is None:
            raise RuntimeError("ECODScorer.score called before fit")
        # PyOD's decision_function returns higher = more anomalous (good)
        return np.asarray(self._model.decision_function(np.asarray(X, dtype=np.float64))).flatten()


# ---------------------------------------------------------------------------
# Factory: build all scorers from one place so the sweep is configuration-driven
# ---------------------------------------------------------------------------

def build_default_scorers():
    """Return the full scorer roster. Sweep iterates over these by name."""
    return {
        "bnn_mean": BNNMeanScorer(),
        "bnn_variance": BNNVarianceScorer(),
        "bnn_combined": BNNCombinedScorer(),
        "isolation_forest": IsolationForestScorer(),
        "autoencoder": AutoencoderScorer(),
        "ecod": ECODScorer(),
    }


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(0)
    n_norm, n_anom = 800, 80
    X_train = np.vstack([
        rng.normal(0, 1, size=(n_norm, 8)),
        rng.normal(3, 1, size=(n_anom, 8)),
    ]).astype(np.float32)
    y_train = np.array([0] * n_norm + [1] * n_anom)

    n_te_norm, n_te_anom = 200, 40
    X_test = np.vstack([
        rng.normal(0, 1, size=(n_te_norm, 8)),
        rng.normal(3, 1, size=(n_te_anom, 8)),
    ]).astype(np.float32)
    y_test = np.array([0] * n_te_norm + [1] * n_te_anom)

    print(f"{'scorer':<22} {'AUC-like (sep)':>15} {'mean_norm':>12} {'mean_anom':>12}")
    for name, s in build_default_scorers().items():
        if name.startswith("bnn"):
            continue  # skip BNN scorers in self-test (need a trained framework)
        s.fit(X_train, y_train)
        scores = s.score(X_test)
        sep = scores[y_test == 1].mean() - scores[y_test == 0].mean()
        print(f"{name:<22} {sep:>15.4f} {scores[y_test==0].mean():>12.4f} {scores[y_test==1].mean():>12.4f}")
