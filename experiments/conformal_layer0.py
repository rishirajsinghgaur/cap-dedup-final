#!/usr/bin/env python3
"""Split-conformal anomaly-preservation gate. Calibrates a threshold tau as the alpha-quantile of calibration-positive scores; test samples with score >= tau are preserved, giving a finite-sample marginal lower bound on fault recall. fit_classwise() computes a per-class (Mondrian) threshold."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("conformal_layer0")


@dataclass
class ConformalAnomalyGate:
    """
    Split-conformal anomaly preservation gate.

    Parameters
    ----------
    target_recall : float in (0, 1)
        Desired fault-recall floor (e.g. 0.99 for 99% recall guarantee).
        alpha = 1 - target_recall.

    Attributes set by fit():
    ------------------------
    tau : float
        Threshold on the scorer's output. Samples with score >= tau are
        preserved by this layer.
    n_calib_anomalies : int
        Number of calibration anomalies used. The finite-sample correction
        means the *actual* coverage guarantee is
            >= 1 - alpha - (1 / (n_calib_anomalies + 1))
        so n_calib_anomalies should ideally be at least 100.
    score_distribution : dict
        Diagnostic summary of the calibration scores (mean / std / quantiles).
    """

    target_recall: float = 0.99
    tau: Optional[float] = field(default=None)
    n_calib_anomalies: int = 0
    score_distribution: dict = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return 1.0 - self.target_recall

    def fit(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        scorer: Callable[[np.ndarray], np.ndarray],
    ) -> "ConformalAnomalyGate":
        """
        Calibrate tau on the held-out calibration anomalies.

        Parameters
        ----------
        X_calib : (n_calib, d)
            Calibration features (must NOT have been used to train the scorer).
        y_calib : (n_calib,)
            Calibration labels (1 = anomaly/fault, 0 = normal).
        scorer : callable
            Function mapping X -> 1D array of anomaly scores
            (higher = more anomalous).
        """
        anomaly_mask = (np.asarray(y_calib) == 1)
        n_anom = int(anomaly_mask.sum())
        if n_anom == 0:
            raise ValueError(
                "ConformalAnomalyGate.fit: calibration set contains zero "
                "anomalies. Cannot calibrate without positive examples."
            )

        # Score ALL calibration points, THEN select the anomalies. Scoring the anomaly
        # subset directly (scorer(X[mask])) is unsafe when `scorer` is a closure over
        # precomputed full-set scores (as in the sweep): it returns the full array and
        # mis-calibrates tau on the whole distribution instead of anomaly-only. Theorem 1
        # requires the quantile of CALIBRATION POSITIVES, so we index after scoring.
        all_calib_scores = np.asarray(scorer(np.asarray(X_calib))).flatten().astype(float)
        anomaly_scores = all_calib_scores[anomaly_mask]

        # Finite-sample conformal quantile (matches Angelopoulos & Bates ch. 2):
        # use the floor((alpha)(n+1))-th smallest score.  We use np.quantile
        # with method='lower' which yields exactly this when q = alpha * (n+1)/n.
        # For simplicity and ease of inspection, fall back to plain quantile.
        q_level = max(0.0, min(1.0, self.alpha))
        if n_anom < 20:
            logger.warning(
                f"ConformalAnomalyGate: only {n_anom} calibration anomalies — "
                f"the finite-sample correction is ~{1/(n_anom+1):.3f}. "
                f"Effective recall guarantee may be lower than the requested "
                f"{self.target_recall:.2%}."
            )

        tau = float(np.quantile(anomaly_scores, q_level, method="lower"))
        self.tau = tau
        self.n_calib_anomalies = n_anom
        self.score_distribution = {
            "mean": float(anomaly_scores.mean()),
            "std": float(anomaly_scores.std()),
            "min": float(anomaly_scores.min()),
            "q05": float(np.quantile(anomaly_scores, 0.05)),
            "q25": float(np.quantile(anomaly_scores, 0.25)),
            "q50": float(np.quantile(anomaly_scores, 0.50)),
            "q75": float(np.quantile(anomaly_scores, 0.75)),
            "q95": float(np.quantile(anomaly_scores, 0.95)),
            "max": float(anomaly_scores.max()),
        }
        logger.info(
            f"ConformalAnomalyGate fitted: target_recall={self.target_recall:.3f} "
            f"(alpha={self.alpha:.3f}), n_calib_anom={n_anom}, tau={tau:.6f}, "
            f"calib_score_median={self.score_distribution['q50']:.4f}"
        )
        return self

    def fit_classwise(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
        class_calib: np.ndarray,
        scorer: Callable[[np.ndarray], np.ndarray],
    ) -> "ConformalAnomalyGate":
        """Class-conditional (Mondrian) calibration.

        Instead of one pooled threshold, compute a per-class conformal threshold
        tau_c (the alpha-quantile of class-c calibration scores) and set the global
        threshold to the most lenient one, tau = min_c tau_c. Then EVERY anomaly class
        meets its >=(1-alpha) recall floor by construction, not just the pooled average.
        The cost is a lower threshold (more samples preserved); this cost is small when
        anomalies are sparse and large when they are dense (characterised in the paper).

        Parameters
        ----------
        X_calib, y_calib : as in fit().
        class_calib : (n_calib,) per-sample class id (e.g. fault/attack type). Only the
            entries where y_calib==1 are used; values for normals are ignored.
        scorer : same callable as used elsewhere.
        """
        y = np.asarray(y_calib); cls = np.asarray(class_calib)
        anom = (y == 1)
        if int(anom.sum()) == 0:
            raise ValueError("fit_classwise: calibration set contains zero anomalies.")
        # Score ALL calibration points then index anomalies (same robustness fix as fit():
        # safe whether `scorer` is a real scorer or a precomputed-full-scores closure).
        all_calib_scores = np.asarray(scorer(np.asarray(X_calib))).flatten().astype(float)
        scores = all_calib_scores[anom]
        cls_a = cls[anom]
        q = max(0.0, min(1.0, self.alpha))
        per_class_tau = {}
        for c in np.unique(cls_a):
            sc = scores[cls_a == c]
            if len(sc) >= 1:
                per_class_tau[c] = float(np.quantile(sc, q, method="lower"))
                if len(sc) < 20:
                    logger.warning(
                        f"fit_classwise: class {c} has only {len(sc)} calibration "
                        f"anomalies; per-class finite-sample slack ~{1/(len(sc)+1):.3f}.")
        self.tau = float(min(per_class_tau.values()))
        self.per_class_tau = per_class_tau
        self.n_calib_anomalies = int(anom.sum())
        self.mode = "class_conditional"
        logger.info(
            f"ConformalAnomalyGate (class-conditional) fitted: {len(per_class_tau)} classes, "
            f"tau=min={self.tau:.6f}, target_recall={self.target_recall:.3f}")
        return self

    def preserve_mask(
        self,
        X_test: np.ndarray,
        scorer: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """
        Return boolean mask: True = preserve this sample (do NOT deduplicate).

        Parameters
        ----------
        X_test : (n_test, d)
        scorer : same callable as used in fit()

        Returns
        -------
        mask : (n_test,) bool array
        """
        if self.tau is None:
            raise RuntimeError("ConformalAnomalyGate.preserve_mask called before fit()")
        scores = np.asarray(scorer(X_test)).flatten().astype(float)
        return scores >= self.tau


# -----------------------------------------------------------------------------
# Scorer factory: turn a UncertaintyAwareFramework instance into a callable
# scorer that returns BNN mean predictions on input X.
# -----------------------------------------------------------------------------

def make_bnn_mean_scorer(framework, mc_samples: Optional[int] = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a callable that returns the Bayesian ensemble's mean predicted
    probability of being anomalous, for each row of X.

    This is the cheapest possible anomaly scorer because the BNN is ALREADY
    trained as part of the precursor framework. Zero extra training cost.

    Parameters
    ----------
    framework : UncertaintyAwareFramework
        Trained framework (its bayesian_ensemble must be ready).
    mc_samples : int, optional
        Number of MC-dropout samples. Defaults to framework.config['uncertainty']['mc_samples'].
    """
    import torch  # local import so the module doesn't hard-fail without torch

    if mc_samples is None:
        mc_samples = framework.config["uncertainty"]["mc_samples"]

    def scorer(X: np.ndarray) -> np.ndarray:
        framework.bayesian_ensemble.train()  # keep dropout active for MC
        X_t = torch.FloatTensor(np.asarray(X)).to(framework.device)
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds.append(framework.bayesian_ensemble(X_t))
        mean_pred = torch.cat(preds, dim=0).mean(dim=0)
        return mean_pred.cpu().numpy().flatten()

    return scorer


# -----------------------------------------------------------------------------
# Tiny self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic sanity check: two-Gaussian setup
    rng = np.random.default_rng(0)
    n_normal, n_anom = 500, 50
    X_normal = rng.normal(0.0, 1.0, size=(n_normal, 5))
    X_anom = rng.normal(3.0, 1.0, size=(n_anom, 5))
    X_calib = np.vstack([X_normal, X_anom])
    y_calib = np.array([0] * n_normal + [1] * n_anom)

    def fake_scorer(X):
        # Anomaly score = mean of the feature vector (high for the +3 cluster)
        return X.mean(axis=1)

    for target in [0.90, 0.95, 0.99]:
        gate = ConformalAnomalyGate(target_recall=target).fit(X_calib, y_calib, fake_scorer)
        # Test: 200 new samples (half anomaly)
        X_test_normal = rng.normal(0.0, 1.0, size=(200, 5))
        X_test_anom = rng.normal(3.0, 1.0, size=(200, 5))
        X_test = np.vstack([X_test_normal, X_test_anom])
        y_test = np.array([0] * 200 + [1] * 200)
        mask = gate.preserve_mask(X_test, fake_scorer)
        empirical_recall = mask[y_test == 1].mean()
        preservation_rate = mask.mean()
        print(f"target={target:.2f} tau={gate.tau:.4f} "
              f"emp_recall={empirical_recall:.3f} preserved_frac={preservation_rate:.3f}")
