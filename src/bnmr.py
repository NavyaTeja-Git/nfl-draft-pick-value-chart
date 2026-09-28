"""Bayesian nonparametric monotone regression with Bernstein polynomials.

Python implementation of Wilson, Tryner, L'Orange and Volckens (2020),
"Bayesian nonparametric monotone regression", Environmetrics 31(8), e2642
(arXiv:2006.00326), used here for a monotone DEcreasing draft value curve.

Model (paper section 2.1), for x in [0, 1] and f increasing in x:
    y_i = f(x_i) + e_i,             e_i ~ N(0, sigma^2)
    f(x) = sum_k psi_k(x, M) beta_k,  psi_k = Bernstein basis of order M
    theta_0 = beta_0,  theta_k = beta_k - beta_{k-1}   (k = 1..M)
    f is increasing when theta_k >= 0 for all k >= 1, and
    f(x) = sum_k Lambda_k(x) theta_k  with  Lambda_k(x) = sum_{j>=k} psi_j(x, M)
         = P(Binomial(M, x) >= k).
Prior:
    theta_k | G ~ G,   G ~ DP(alpha G0),   G0 = pi delta_0 + (1 - pi) TN_[0,inf)(mu, phi^2)
    pi ~ Beta(a_pi, b_pi) (integrated out),  alpha ~ Gamma(a_alpha, b_alpha)
    theta_0 ~ N(0, phi0^2),  sigma^-2 ~ Gamma(a_sigma, b_sigma)
y is standardized (mean 0, sd 1) and mu = 0.5, phi = 0.25 as in the paper (section 2.3).

Sampler (section 2.2): Polya-urn Gibbs updates of each theta_k's cluster
(zero cluster, an existing positive cluster, or a new one), then the intercept
and all cluster values jointly as one truncated multivariate normal block
(exact rejection sampling, with a Gibbs fallback instead of the Li and Ghosh
2015 sampler), then sigma^2 and alpha (Escobar & West 1995).

Because f depends on the data only through x, the likelihood is computed from
per-x sufficient statistics (count, sum of y, sum of y^2), which is exact.

For a decreasing curve in pick d, use x = 1 - (d - 1) / (d_max - 1).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import binom, truncnorm


def lambda_basis(x: np.ndarray, M: int) -> np.ndarray:
    """n x (M+1) matrix: column 0 is 1, column k is P(Binomial(M, x) >= k)."""
    k = np.arange(M + 1)
    return binom.sf(k[None, :] - 1, M, np.asarray(x, float)[:, None])


@dataclass
class Prior:
    M: int = 50
    mu: float = 0.5
    phi: float = 0.25
    phi0: float = 10.0          # sd of the intercept prior (standardized scale)
    a_pi: float = 1.0
    b_pi: float = 1.0
    a_alpha: float = 1.0
    b_alpha: float = 1.0
    a_sigma: float = 0.1
    b_sigma: float = 0.1


@dataclass
class State:
    theta: np.ndarray          # length M+1, theta[0] = intercept
    labels: np.ndarray         # length M+1, labels[0] unused; 0 = zero cluster, c >= 1 = positive cluster
    eta: dict = field(default_factory=dict)   # cluster label -> value
    sigma2: float = 1.0
    alpha: float = 1.0


def _truncnorm_pos(mean: float, var: float, rng) -> float:
    sd = np.sqrt(var)
    return float(truncnorm.rvs(-mean / sd, np.inf, loc=mean, scale=sd, random_state=rng))


def _truncated_mvn(P, h, x_current, rng, tries=20, inner=30):
    """Draw x ~ N(P^-1 h, P^-1) restricted to x[1:] > 0 (x[0] unrestricted).

    First tries exact rejection sampling from the unrestricted normal. If the
    restricted region has low probability, falls back to `inner` sweeps of
    one-coordinate-at-a-time Gibbs started from the current (valid) value,
    which leaves the same distribution invariant.
    """
    chol = np.linalg.cholesky(P)                      # P = chol @ chol.T
    mean = np.linalg.solve(P, h)
    for _ in range(tries):
        x = mean + np.linalg.solve(chol.T, rng.standard_normal(len(h)))
        if np.all(x[1:] > 0):
            return x
    x = np.array(x_current, float)
    for _ in range(inner):
        for j in range(len(x)):
            cond_var = 1.0 / P[j, j]
            cond_mean = mean[j] - cond_var * (P[j] @ (x - mean) - P[j, j] * (x[j] - mean[j]))
            x[j] = rng.normal(cond_mean, np.sqrt(cond_var)) if j == 0 else _truncnorm_pos(cond_mean, cond_var, rng)
    return x


class MonotoneBP:
    """Fit y ~ f(x) with f increasing, from grouped data."""

    def __init__(self, x_unique, n, sum_y, sum_y2, prior: Prior | None = None):
        self.p = prior or Prior()
        self.L = lambda_basis(x_unique, self.p.M)              # G x (M+1)
        self.n = np.asarray(n, float)
        self.N = self.n.sum()
        # standardize y using the grouped totals
        self.ybar = np.sum(sum_y) / self.N
        var = np.sum(sum_y2) / self.N - self.ybar ** 2
        self.ysd = np.sqrt(var)
        self.S = (np.asarray(sum_y) - self.n * self.ybar) / self.ysd                    # sum of z per group
        self.Q = (np.asarray(sum_y2) - 2 * self.ybar * np.asarray(sum_y)
                  + self.n * self.ybar ** 2) / self.ysd ** 2                             # sum of z^2 per group
        self.nL2 = (self.n[:, None] * self.L ** 2).sum(axis=0)                          # sum_i Lambda_ik^2

    # ---- helpers on the standardized scale ----
    def _fitted(self, theta):
        return self.L @ theta

    def _sse(self, f):
        return float(np.sum(self.Q - 2 * f * self.S + self.n * f ** 2))

    def init_state(self, rng) -> State:
        M = self.p.M
        labels = np.zeros(M + 1, int)
        theta = np.zeros(M + 1)
        # start with every increment in one positive cluster (a linear start)
        labels[1:] = 1
        eta = {1: self.p.mu / M}
        theta[1:] = eta[1]
        theta[0] = (self.S.sum() - (self.n * (self.L[:, 1:] @ theta[1:])).sum()) / self.N
        return State(theta=theta, labels=labels, eta=eta, sigma2=1.0, alpha=1.0)

    def sweep(self, st: State, rng) -> State:
        p, M = self.p, self.p.M
        f = self._fitted(st.theta)
        s2 = st.sigma2
        logc_prior_pos = log_ndtr(p.mu / p.phi)

        # 1. cluster labels for theta_1..theta_M (Polya urn)
        for k in range(1, M + 1):
            Lk = self.L[:, k]
            f_minus = f - Lk * st.theta[k]
            b = float(np.sum(Lk * (self.S - self.n * f_minus))) / s2   # sum Lambda_ik r_i / s2
            a = self.nL2[k] / s2                                       # sum Lambda_ik^2 / s2
            old = st.labels[k]
            counts = {}
            for kk in range(1, M + 1):
                if kk != k:
                    counts[st.labels[kk]] = counts.get(st.labels[kk], 0) + 1
            n0 = counts.pop(0, 0)
            n_pos = M - 1 - n0
            w_zero = np.log(n0 + p.a_pi) - np.log(M - 1 + p.a_pi + p.b_pi)
            w_nonzero = np.log(n_pos + p.b_pi) - np.log(M - 1 + p.a_pi + p.b_pi)
            names, logw = [0], [w_zero]                                  # log L(theta=0) is the reference
            for c, nc in counts.items():
                e = st.eta[c]
                names.append(c)
                logw.append(w_nonzero + np.log(nc) - np.log(n_pos + st.alpha) + b * e - 0.5 * a * e * e)
            v = 1.0 / (a + 1.0 / p.phi ** 2)
            m = v * (b + p.mu / p.phi ** 2)
            log_marg = (0.5 * np.log(v) - np.log(p.phi) + m * m / (2 * v) - p.mu ** 2 / (2 * p.phi ** 2)
                        + log_ndtr(m / np.sqrt(v)) - logc_prior_pos)
            names.append(-1)
            logw.append(w_nonzero + np.log(st.alpha) - np.log(n_pos + st.alpha) + log_marg)
            logw = np.array(logw)
            prob = np.exp(logw - logw.max())
            choice = names[rng.choice(len(names), p=prob / prob.sum())]
            if choice == -1:
                new = max(list(st.eta) + [0]) + 1
                st.eta[new] = _truncnorm_pos(m, v, rng)
                choice = new
            st.labels[k] = choice
            st.theta[k] = 0.0 if choice == 0 else st.eta[choice]
            f = f_minus + Lk * st.theta[k]
            if old != 0 and old != choice and not np.any(st.labels[1:] == old):
                del st.eta[old]

        # 2-3. intercept and all positive cluster values jointly (paper section 2.2):
        # eta = (theta_0, eta_1..eta_K) has a multivariate normal full conditional,
        # truncated to eta_c > 0. They are strongly correlated (the intercept and the
        # increments can both shift the curve), so they are sampled as one block.
        clusters = list(st.eta)
        U = np.column_stack([np.ones(len(self.n))] +
                            [self.L[:, [k for k in range(1, M + 1) if st.labels[k] == c]].sum(axis=1)
                             for c in clusters])
        P = (U * self.n[:, None]).T @ U / s2
        P[np.diag_indices_from(P)] += np.r_[1.0 / p.phi0 ** 2, np.full(len(clusters), 1.0 / p.phi ** 2)]
        h = U.T @ self.S / s2 + np.r_[0.0, np.full(len(clusters), p.mu / p.phi ** 2)]
        x = _truncated_mvn(P, h, np.r_[st.theta[0], [st.eta[c] for c in clusters]], rng)
        st.theta[0] = x[0]
        for c, v in zip(clusters, x[1:]):
            st.eta[c] = v
            st.theta[st.labels == c] = v
        f = U @ x

        # 4. error variance
        st.sigma2 = 1.0 / rng.gamma(p.a_sigma + self.N / 2, 1.0 / (p.b_sigma + self._sse(f) / 2))

        # 5. DP concentration (Escobar & West 1995)
        K = len(st.eta)
        n_obs = int(np.sum(st.labels[1:] != 0))
        if n_obs > 0 and K > 0:
            e = rng.beta(st.alpha + 1, n_obs)
            odds = (p.a_alpha + K - 1) / (n_obs * (p.b_alpha - np.log(e)))
            shape = p.a_alpha + K if rng.random() < odds / (1 + odds) else p.a_alpha + K - 1
            st.alpha = rng.gamma(shape, 1.0 / (p.b_alpha - np.log(e)))
        else:
            st.alpha = rng.gamma(p.a_alpha, 1.0 / p.b_alpha)
        return st

    def run(self, n_burn, n_keep, rng, state: State | None = None, thin=1):
        """Returns (theta draws on the standardized scale, sigma draws on the y scale, final state)."""
        st = state or self.init_state(rng)
        thetas, sigmas = [], []
        for it in range(n_burn + n_keep * thin):
            st = self.sweep(st, rng)
            if it >= n_burn and (it - n_burn) % thin == 0:
                thetas.append(st.theta.copy())
                sigmas.append(np.sqrt(st.sigma2) * self.ysd)
        return np.array(thetas), np.array(sigmas), st

    def curve(self, thetas, x_grid):
        """Posterior draws of f(x_grid) on the original y scale (draws x grid)."""
        L = lambda_basis(x_grid, self.p.M)
        return self.ybar + self.ysd * (thetas @ L.T)
