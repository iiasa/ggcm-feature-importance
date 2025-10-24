import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from typing import Literal, Optional
import os
import json
from scipy.linalg import lstsq


class OOLS():

    def __init__(self, formula: str = None, fit_method: Literal['inv', 'pinv', 'scipy'] = 'inv') -> None:
        """
        Online Ordinary Least-Squares model for streamable fitting to data. 
        """
        self.n = 0
        self.p = 0
        self._fit_method = fit_method
        self.c: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.formula = formula
        self._ssr = 0.

        self.xi: Optional[np.ndarray] = None
        self.stats: dict = None

    def fit(self, x: ArrayLike, y: ArrayLike):
        self.n += x.shape[0]
        self.p = x.shape[1]
        if self._fit_method != 'scipy':
            self.c = np.linalg.inv(x.T @ x) if self._fit_method == 'inv' else np.linalg.pinv(x.T @ x)
            self.beta = self.c @ x.T @ y
        else:
            self.c = np.linalg.pinv(x.T @ x)  # dodgy
            self.beta, _, _, s = lstsq(x, y)
        res = y - x @ self.beta
        self._ssr = res.T @ res

    def partial_fit(self, x: ArrayLike, y: ArrayLike):
        """The online fitting function is base on the Sherman-Morrison-Woodbury (SMW) identity. 
        (A+UCV)^{-1} = A^{-1}-A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}, with U=V=I is:
        (A+C)^{-1} = A^{-1}-A^{-1}(C^{-1}+A^{-1})^{-1}A^{-1}, with A=(x_0'x_0)^{-1} and C=(x_1'x_1)^{-1}
        Note that this solution is tailored to large N in x_1, i.e., the allocation of a NxN matrix is avoided. 
        For small N a different approach might be more efficient, see [1].
        [1] "Streaming solutions to least-squares problems", Georgia Tech, lecture notes
        """
        self.n += x.shape[0]
        self.p = x.shape[1]

        xxi = np.linalg.inv(x.T @ x)

        if self.c is None:
            self.c = xxi
            self.beta = self.c @ x.T @ y
        else:
            self.c = self.c - self.c @ np.linalg.inv(xxi + self.c) @ self.c
            k = self.c @ x.T
            self.beta = self.beta + k @ (y - x @ self.beta)

        # Operating with with xi instead of xxi and a pseudoinverse would improve numerical stability and
        # performance when calculating, e.g., the standard error of the fit. However, as fitting time is 
        # more than 4 times slower we stay with xxi for now.

        # xi = np.linalg.pinv(x)
        # if self.xi is None:
        #     self.xi = xi
        #     self.beta = self.xi @ self.xi.T @ x.T @ y
        # else:
        #     self.xi = self.xi - self.xi @ np.linalg.pinv(xi + self.xi) @ self.xi
        #     k = self.xi @ self.xi.T @ x.T
        #     self.beta = self.beta + k @ (y - x @ self.beta)

        # Approaches to a "cumulative SSR":
        # (i) Add SSR of the current fit to the overall SSR. This overestimates the SSR because
        # the regression line improves with every fit and residuals from previous fits are no longer 
        # valid.
        # (ii) Use only the latest fit for SSR computations, whereas the degrees of freedom should be 
        # reduced to last fit's n. This probably also overestimates SSR, as a lot of DF are 
        # sacrificed in the MSE calculation.
        # Generally, either should be fine as both methods are conservative in terms of error-estimation.
        res = y - x @ self.beta
        self._ssr += res.T @ res

    @property
    def mse(self) -> float:
        """
        Mean squared error of the training data; estimator for \sigma^2.
        """
        return self._ssr / (self.n - self.p)

    def se2(self, x: ArrayLike) -> np.ndarray:
        """
        Squared standard error of the fit for x (slow); sqrt for standard error.
        """
        return self.mse * np.apply_along_axis(lambda x_: x_ @ self.c @ x_.T, 1, x)

    def conf_int(self, x: ArrayLike, alpha: float) -> np.ndarray:
        """
        Returns confidence interval half-widths for each row in x.
        """
        t = stats.t.ppf(1 - alpha / 2, self.n - self.p)
        return t * np.sqrt(self.se2(x))

    def pred_int(self, x: ArrayLike, alpha: float) -> np.ndarray:
        """
        Returns prediction interval half-widths for each row in x.
        """
        t = stats.t.ppf(1 - alpha / 2, self.n - self.p)
        return t * np.sqrt(self.mse + self.se2(x))

    def predict(self, x: ArrayLike) -> np.ndarray:
        return np.array(x) @ self.beta

    def to_dict(self, stats: dict = None) -> dict:
        return {
            'stats': stats or {},
            'beta': self.beta.tolist(),
            'c': self.c.tolist(),
            'n': self.n,
            'p': self.p,
            'formula': self.formula,
            'ssr': self._ssr
        }
    
    @staticmethod
    def from_dict(d: dict) -> 'OOLS':
        ools = OOLS()
        ools.beta = np.array(d['beta'])
        ools.c = np.array(d['c'])
        ools.n = d['n']
        ools.p = d['p']
        ools.formula = d['formula']
        ools._ssr = d['ssr']
        if 'stats' in d:
            ools.stats = d['stats']
        return ools

    def save(self, filename_json: str, stats: dict = None):
        with open(filename_json, 'w') as file:
            json.dump(self.to_dict(stats), file)

    @staticmethod
    def load(filename_json: str) -> 'OOLS':
        with open(filename_json, 'r') as file:
            return OOLS.from_dict(json.load(file))
    
    def calc_stats_from_res(self, res: np.array, yss: np.array):
        n = len(res)
        ssr = res.T @ res
        mse = ssr / (n - self.p)
        sst = yss.T @ yss
        r2 = 1 - ssr / sst
        r2_adj = 1 - (ssr / (n - self.p)) / (sst / (n - 1))

        return {
            'r2': r2,
            'r2_adj': r2_adj,
            'mse': mse, 
            'n': n,
            'p': self.p
        }
