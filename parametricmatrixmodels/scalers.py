from typing import Optional, Tuple, Union

import jax.numpy as np

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
