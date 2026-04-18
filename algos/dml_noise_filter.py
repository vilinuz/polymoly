"""
Double/Debiased Machine Learning (DML) Noise Isolation Engine.

Implements the Chernozhukov et al. (2018) DML framework to extract
clean causal signals from high-dimensional, noisy prediction market data.

Architecture:
  - K-fold cross-fitting to eliminate overfitting bias
  - Flexible first-stage nuisance estimators (RF, GBM, Lasso, ElasticNet)
  - Neyman-orthogonal score construction
  - Confidence intervals and hypothesis testing for treatment effects
  - Online re-estimation as new data streams in
  - Integration with Bayesian forecaster outputs as treatment variables
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger("DMLNoiseFilter")


@dataclass
class DMLEstimate:
    """Container for a single DML estimation result."""
    theta: float                    # point estimate of causal effect
    se: float                       # standard error
    ci_lower: float                 # 95% CI lower
    ci_upper: float                 # 95% CI upper
    t_stat: float
    p_value: float
    n_obs: int
    n_folds: int
    first_stage_r2_y: float         # R² of nuisance model for outcome
    first_stage_r2_t: float         # R² of nuisance model for treatment
    timestamp: float = field(default_factory=time.time)

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


@dataclass
class CATEEstimate:
    """Conditional Average Treatment Effect across covariate groups."""
    group_effects: Dict[str, float]     # group_name → CATE
    group_ses: Dict[str, float]
    group_cis: Dict[str, Tuple[float, float]]
    heterogeneity_pvalue: float         # test for effect heterogeneity


class DMLNoiseFilter:
    """
    Double/Debiased Machine Learning engine for causal effect estimation
    in high-dimensional prediction market settings.

    Treatment variable (T): The causal variable of interest, e.g.
        daily poll shifts, endorsement announcements, debate outcomes.
    Outcome variable (Y): The target, e.g. Polymarket price changes,
        log returns, implied probability shifts.
    Covariates (X): High-dimensional confounders including volume,
        spread, macro indicators, sentiment scores, etc.

    The DML procedure:
      1. Estimate E[Y|X] using ML model → residualize Y
      2. Estimate E[T|X] using ML model → residualize T
      3. Regress residualized Y on residualized T → θ (debiased effect)
    """

    def __init__(
        self,
        n_folds: int = 5,
        model_y: str = "gradient_boosting",
        model_t: str = "lasso",
        n_repetitions: int = 3,
    ):
        self.n_folds = n_folds
        self.n_repetitions = n_repetitions
        self._model_y_name = model_y
        self._model_t_name = model_t
        self._scaler_x = StandardScaler()
        self._scaler_t = StandardScaler()
        self._last_estimate: Optional[DMLEstimate] = None
        self._estimate_history: List[DMLEstimate] = []

    # ─── Model Factory ──────────────────────────────────────────────────

    @staticmethod
    def _make_model(name: str):
        factories = {
            "random_forest": lambda: RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                n_jobs=-1, random_state=42,
            ),
            "gradient_boosting": lambda: GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
            "lasso": lambda: LassoCV(cv=5, random_state=42, max_iter=5000),
            "elastic_net": lambda: ElasticNetCV(cv=5, random_state=42, max_iter=5000),
        }
        if name not in factories:
            raise ValueError(f"Unknown model: {name}. Choose from {list(factories.keys())}")
        return factories[name]()

    # ─── Core DML Estimation ────────────────────────────────────────────

    def fit(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
    ) -> DMLEstimate:
        """
        Run the full DML procedure with K-fold cross-fitting.

        Args:
            Y: (n,) outcome vector
            T: (n,) treatment vector
            X: (n, p) covariate matrix

        Returns:
            DMLEstimate with point estimate, SE, CI, and diagnostics.
        """
        n = len(Y)
        Y = np.asarray(Y, dtype=np.float64).ravel()
        T = np.asarray(T, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)

        assert Y.shape[0] == T.shape[0] == X.shape[0], "Dimension mismatch"

        # Standardize covariates
        X_scaled = self._scaler_x.fit_transform(X)

        logger.info(
            f"Running DML: n={n}, p={X.shape[1]}, "
            f"folds={self.n_folds}, reps={self.n_repetitions}"
        )

        # Multiple-repetition DML for stability
        thetas = []
        all_r2_y, all_r2_t = [], []

        for rep in range(self.n_repetitions):
            theta_rep, r2_y, r2_t = self._single_dml_pass(
                Y, T, X_scaled, random_state=rep * 42
            )
            thetas.append(theta_rep)
            all_r2_y.append(r2_y)
            all_r2_t.append(r2_t)

        # Median aggregation across repetitions (Chernozhukov et al.)
        theta = float(np.median(thetas))

        # Standard error via influence function approach
        se = self._compute_se_influence(Y, T, X_scaled, theta)

        # Confidence interval and test
        z = 1.96
        ci_lower = theta - z * se
        ci_upper = theta + z * se
        t_stat = theta / se if se > 1e-12 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

        estimate = DMLEstimate(
            theta=theta,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            t_stat=t_stat,
            p_value=p_value,
            n_obs=n,
            n_folds=self.n_folds,
            first_stage_r2_y=float(np.mean(all_r2_y)),
            first_stage_r2_t=float(np.mean(all_r2_t)),
        )

        self._last_estimate = estimate
        self._estimate_history.append(estimate)

        logger.info(
            f"DML Result: θ={theta:.6f}, SE={se:.6f}, "
            f"CI=[{ci_lower:.6f}, {ci_upper:.6f}], p={p_value:.4f}, "
            f"R²_y={np.mean(all_r2_y):.3f}, R²_t={np.mean(all_r2_t):.3f}"
        )
        return estimate

    def _single_dml_pass(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        random_state: int = 42,
    ) -> Tuple[float, float, float]:
        """One pass of K-fold cross-fitted DML."""
        n = len(Y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=random_state)

        # Residual containers
        res_Y = np.zeros(n)
        res_T = np.zeros(n)
        r2_y_scores, r2_t_scores = [], []

        model_y = self._make_model(self._model_y_name)
        model_t = self._make_model(self._model_t_name)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]

            # First stage: E[Y|X]
            my = clone(model_y).fit(X_train, Y_train)
            Y_hat = my.predict(X_test)
            res_Y[test_idx] = Y_test - Y_hat

            # R² for Y model
            ss_res = np.sum((Y_test - Y_hat) ** 2)
            ss_tot = np.sum((Y_test - Y_test.mean()) ** 2)
            r2_y_scores.append(1 - ss_res / (ss_tot + 1e-12))

            # First stage: E[T|X]
            mt = clone(model_t).fit(X_train, T_train)
            T_hat = mt.predict(X_test)
            res_T[test_idx] = T_test - T_hat

            # R² for T model
            ss_res_t = np.sum((T_test - T_hat) ** 2)
            ss_tot_t = np.sum((T_test - T_test.mean()) ** 2)
            r2_t_scores.append(1 - ss_res_t / (ss_tot_t + 1e-12))

        # Second stage: OLS of residualized Y on residualized T
        # θ = (V̂'V̂)^{-1} V̂'Û
        theta = np.dot(res_T, res_Y) / (np.dot(res_T, res_T) + 1e-12)

        return theta, float(np.mean(r2_y_scores)), float(np.mean(r2_t_scores))

    def _compute_se_influence(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        theta: float,
    ) -> float:
        """
        Compute standard error using Neyman-orthogonal influence function.
        ψ_i = (Y_i - E[Y|X_i] - θ(T_i - E[T|X_i])) * (T_i - E[T|X_i])
        SE = sqrt(Var(ψ) / n)
        """
        n = len(Y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)

        model_y = self._make_model(self._model_y_name)
        model_t = self._make_model(self._model_t_name)

        psi = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            my = clone(model_y).fit(X[train_idx], Y[train_idx])
            mt = clone(model_t).fit(X[train_idx], T[train_idx])

            res_y = Y[test_idx] - my.predict(X[test_idx])
            res_t = T[test_idx] - mt.predict(X[test_idx])

            psi[test_idx] = (res_y - theta * res_t) * res_t

        J = np.mean(psi ** 2)
        se = np.sqrt(J / n)
        return float(se)

    # ─── Conditional Average Treatment Effect ───────────────────────────

    def estimate_cate(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        group_col_idx: int,
        group_labels: Dict[int, str],
    ) -> CATEEstimate:
        """
        Estimate heterogeneous treatment effects across subgroups
        defined by a categorical variable in the covariate matrix.
        """
        unique_groups = np.unique(X[:, group_col_idx])
        effects = {}
        ses = {}
        cis = {}

        for gval in unique_groups:
            mask = X[:, group_col_idx] == gval
            if mask.sum() < 30:
                continue
            sub_est = self.fit(Y[mask], T[mask], X[mask])
            label = group_labels.get(int(gval), f"group_{int(gval)}")
            effects[label] = sub_est.theta
            ses[label] = sub_est.se
            cis[label] = (sub_est.ci_lower, sub_est.ci_upper)

        # Test for heterogeneity (Wald test)
        if len(effects) > 1:
            vals = np.array(list(effects.values()))
            se_vals = np.array(list(ses.values()))
            mean_effect = np.mean(vals)
            chi2 = np.sum(((vals - mean_effect) / (se_vals + 1e-12)) ** 2)
            het_pval = float(1 - stats.chi2.cdf(chi2, df=len(vals) - 1))
        else:
            het_pval = 1.0

        return CATEEstimate(
            group_effects=effects,
            group_ses=ses,
            group_cis=cis,
            heterogeneity_pvalue=het_pval,
        )

    # ─── Online Update ──────────────────────────────────────────────────

    def update_incremental(
        self,
        Y_new: np.ndarray,
        T_new: np.ndarray,
        X_new: np.ndarray,
        Y_old: np.ndarray,
        T_old: np.ndarray,
        X_old: np.ndarray,
        window: int = 5000,
    ) -> DMLEstimate:
        """
        Re-estimate the causal effect using a rolling window
        combining historical and new observations.
        """
        Y_combined = np.concatenate([Y_old, Y_new])[-window:]
        T_combined = np.concatenate([T_old, T_new])[-window:]
        X_combined = np.vstack([X_old, X_new])[-window:]
        return self.fit(Y_combined, T_combined, X_combined)

    # ─── Accessors ──────────────────────────────────────────────────────

    @property
    def last_estimate(self) -> Optional[DMLEstimate]:
        return self._last_estimate

    @property
    def estimate_history(self) -> List[DMLEstimate]:
        return self._estimate_history
