"""
3-state Gaussian HMM -> regime TYPE: DIRECTIONAL / RANGING / VOLATILE.
Direction (up/down) is added later by the classifier from trend slope, so the
HMM only has to learn type, which it does far more stably than 4 signed states.

Look-ahead safety: fit on a trailing window ending at the current closed bar and
read the posterior of the LAST observation only. No future data is ever in the
sequence, so the forward-backward smoothing at the final bar reduces to filtering.

State-permutation handling: hmmlearn assigns arbitrary state indices each fit, so
we relabel every fit by per-state statistics (variance picks VOLATILE; |mean| of
the remaining two picks DIRECTIONAL vs RANGING).
"""
from __future__ import annotations
import logging
import warnings
import numpy as np
import pandas as pd

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_OK = True
except Exception:  # pragma: no cover
    _HMM_OK = False

REGIME_TYPES = ("DIRECTIONAL", "RANGING", "VOLATILE")


def _features(close: pd.Series) -> np.ndarray:
    ret = np.log(close).diff()
    rng = ret.abs()  # bar-level realized range proxy
    feat = np.column_stack([ret.values, rng.values])
    return feat


def label_states(model: "GaussianHMM") -> dict[int, str]:
    """Map raw HMM state indices -> regime type by mean/variance."""
    means = model.means_[:, 0]                # mean log-return per state
    variances = np.array([np.trace(c) for c in model.covars_])  # total var per state
    order_var = np.argsort(variances)         # ascending variance
    volatile = int(order_var[-1])             # highest variance -> VOLATILE
    low_two = [s for s in range(model.n_components) if s != volatile]
    # of the two calmer states, larger |mean| is DIRECTIONAL
    directional = max(low_two, key=lambda s: abs(means[s]))
    ranging = [s for s in low_two if s != directional][0]
    return {volatile: "VOLATILE", directional: "DIRECTIONAL", ranging: "RANGING"}


class HMMRegime:
    """Stateful, walk-forward HMM you call bar-by-bar on a trailing window."""

    def __init__(self, fit_window: int = 500, n_states: int = 3, refit_every: int = 25, seed: int = 7):
        if not _HMM_OK:
            raise ImportError("hmmlearn not installed. pip install hmmlearn")
        self.fit_window = fit_window
        self.n_states = n_states
        self.refit_every = refit_every
        self.seed = seed
        self._model = None
        self._labels: dict[int, str] = {}
        self._since_fit = 10 ** 9

    def _fit(self, feat: np.ndarray) -> bool:
        feat = feat[~np.isnan(feat).any(axis=1)]
        if len(feat) < max(50, self.n_states * 10):
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=self.n_states, covariance_type="diag",
                            n_iter=100, tol=1e-2, random_state=self.seed)
            try:
                m.fit(feat)
            except Exception:
                return False
        self._model = m
        self._labels = label_states(m)
        self._since_fit = 0
        return True

    def update(self, close_window: pd.Series) -> dict:
        """close_window: trailing closes ending at the current closed bar."""
        feat_all = _features(close_window)
        feat = feat_all[-self.fit_window:]
        if self._model is None or self._since_fit >= self.refit_every:
            self._fit(feat)
        if self._model is None:
            return {"type": None, "posterior": 0.0}
        self._since_fit += 1
        seq = feat[~np.isnan(feat).any(axis=1)]
        if len(seq) == 0:
            return {"type": None, "posterior": 0.0}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                post = self._model.predict_proba(seq)[-1]   # posterior, last bar
            except Exception:
                return {"type": None, "posterior": 0.0}
        state = int(np.argmax(post))
        return {"type": self._labels.get(state), "posterior": float(post[state])}
