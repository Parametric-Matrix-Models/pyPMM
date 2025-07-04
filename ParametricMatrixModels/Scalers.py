import jax.numpy as np
from typing import Optional, Union, Tuple

"""
StandardScaler and UniformScaler
"""


class UniformScaler:
    """
    Uniform scaler

    Scales all input data to the range [clow, chigh] and all output data to the
    range [Elow, Ehigh]

    PMMs generally work best with [-1, 1] uniform scaling
    """

    def __init__(
        self,
        clow: float,
        chigh: float,
        Elow: Optional[float] = None,
        Ehigh: Optional[float] = None,
    ) -> None:
        self.cmin = None
        self.cmax = None
        self.Emin = None
        self.Emax = None
        self.clow = clow
        self.chigh = chigh
        self.Elow = Elow
        self.Ehigh = Ehigh

    def fit(self, cs: np.ndarray, Es: Optional[np.ndarray] = None) -> None:
        """
        Fit the scaler to the data
        """
        self.cmin = np.min(cs, axis=0, keepdims=True)
        self.cmax = np.max(cs, axis=0, keepdims=True)
        if Es is not None:
            self.Emin = np.min(Es, axis=0, keepdims=True)
            self.Emax = np.max(Es, axis=0, keepdims=True)

    def transform(
        self, cs: np.ndarray, Es: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Transform the data
        """
        scaled_cs = (cs - self.cmin) / (self.cmax - self.cmin) * (
            self.chigh - self.clow
        ) + self.clow

        if Es is None:
            return scaled_cs

        scaled_Es = (Es - self.Emin) / (self.Emax - self.Emin) * (
            self.Ehigh - self.Elow
        ) + self.Elow
        return scaled_cs, scaled_Es

    def inverse_transform(
        self, cs_scaled: np.ndarray, Es_scaled: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Inverse transform the data
        """
        cs = (cs_scaled - self.clow) / (self.chigh - self.clow) * (
            self.cmax - self.cmin
        ) + self.cmin

        if Es_scaled is None:
            return cs

        Es = (Es_scaled - self.Elow) / (self.Ehigh - self.Elow) * (
            self.Emax - self.Emin
        ) + self.Emin
        return cs, Es

    def fit_transform(
        self, cs: np.ndarray, Es: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fit the scaler to the data and transform the data
        """
        self.fit(cs, Es)
        return self.transform(cs, Es)

    # def transform_pmm(self, model: "PMM") -> "PMM":
    #     """
    #     Transform the matrices in the PMM to automatically scale the data
    #     """

    #     if self.cmin is None or self.cmax is None:
    #         raise ValueError("Scaler not fitted yet")
    #     if self.Emin is not None:
    #         assert self.Emin.ndim == 1, "Output scaling only supported for 1D"
    #         assert self.Emax.ndim == 1, "Output scaling only supported for 1D"

    #     n = model.n
    #     p = model.p
    #     A_s = model.A_s
    #     Bs_s = model.Bs_s

    #     # M(c) = A + sum_i B_i * c_i
    #     # so for the c scale
    #     # M_scaled(c) = A + sum_i B_i * [(c_i - mn) / (mx - mn) * (h - l) + l]
    #     #   = [A + sum_i (l - (mn / (mx - mn))) * B_i]
    #     #        + sum_i [B_i * (h - l) / (mx - mn)] c_i
    #     cmin_ = self.cmin[0]
    #     cmax_ = self.cmax[0]
    #     clow_ = self.clow
    #     chigh_ = self.chigh

    #     A_offset = np.einsum(
    #         "i,qijk->qjk", clow_ - cmin_ / (cmax_ - cmin_), Bs_s
    #     )
    #     A_s_scaled = A_s + A_offset

    #     Bs_s_scaled = np.einsum(
    #         "i,qijk->qijk", (chigh_ - clow_) / (cmax_ - cmin_), Bs_s
    #     )

    #     # for the E scale
    #     # M(c) -> M(c) * (h - l) / (mx - mn) + l * ident(n)
    #     # this only works if Es is Nx1
    #     if self.Emin is not None:
    #         Emin_ = self.Emin[0]
    #         Emax_ = self.Emax[0]
    #         Elow_ = self.Elow
    #         Ehigh_ = self.Ehigh

    #         A_s_scaled *= (Ehigh_ - Elow_) / (Emax_ - Emin_)
    #         Bs_s_scaled *= (Ehigh_ - Elow_) / (Emax_ - Emin_)
    #         A_s_scaled += Elow_ * np.eye(n)

    #         # if the nonconformity scores is not None, then scale them directly
    #         if model.nonconformity_scores is not None:
    #             # nonconformity = |E_true - E_pred|
    #             # so we the Elow_ offsets cancel
    #             model._nonconformity_scores = (
    #                 model.nonconformity_scores
    #                 * (Ehigh_ - Elow_)
    #                 / (Emax_ - Emin_)
    #             )

    #     # update the model
    #     model._A_s = A_s_scaled
    #     model._Bs_s = Bs_s_scaled

    #     return model
