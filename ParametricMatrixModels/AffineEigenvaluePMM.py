from .BaseModule import BaseModule
import jax.numpy as np
import jax
from typing import Callable, Tuple, Any, Union, Optional, Dict
from ._affine_backing_funcs import affine_pmm_predict_func


class AffineEigenvaluePMM(BaseModule):
    """
    AffineEigenvaluePMM is a module that implements the affine eigenvalue
    Parametric Matrix Model (PMM).
    """

    def __init__(
        self,
        n: int = None,
        k: int = 1,
        which: str = "SA",
        Ms: Optional[np.ndarray] = None,
        init_magnitude: float = 1e-2,
    ) -> None:
        """
        Initialize the AffineEigenvaluePMM module. Represents a PMM which
        returns the selected eigenvalues of a parametric Hamiltonian matrix

        M(x) = M0 + x1 * M1 + ... + xp * Mp

        By default this module is initialized to compute the smallest algebraic
        eigenvalue (ground state).

        Parameters
        ----------
            n : int
                Size of the PMM matrices (n x n).
            k : int, optional
                Number of eigenvalues to compute, by default 1.
            which : str, optional
                Which eigenvalues to compute, by default "SA".
                Options are:
                - 'SA' for smallest algebraic (default)
                - 'LA' for largest algebraic
                - 'SM' for smallest magnitude
                - 'LM' for largest magnitude
                - 'EA' for exterior algebraically
                - 'EM' for exterior by magnitude
                - 'IA' for interior algebraically
                - 'IM' for interior by magnitude

                For algebraic 'which' options, the eigenvalues are returned in
                ascending algebraic order.

                For magnitude 'which' options, the eigenvalues are returned in
                ascending magnitude order.
            Ms : Optional[np.ndarray], optional
                Optional (p+1, n, n) array of matrices M0, M1, ..., Mp for the
                matrices in the PMM. If not provided, they will be randomly
                initialized when the module is compiled.
            init_magnitude : float, optional
                Initial magnitude for the random matrices if Ms is not
                provided, by default 1e-2.
        """

        # input validation
        if n is None:
            # module will be configured later, hopefully
            self.n = None
            self.k = None
            self.p = None
            self.which = None
            self.Ms = None
            self.init_magnitude = None
            return

        if n <= 0 or not isinstance(n, int):
            raise ValueError("n must be a positive integer")
        if k <= 0 or not isinstance(k, int):
            raise ValueError("k must be a positive integer")
        if k > n:
            raise ValueError("k must be less than or equal to n")
        if which.lower() not in [
            "sa",
            "la",
            "sm",
            "lm",
            "ea",
            "em",
            "ia",
            "im",
        ]:
            raise ValueError(
                f"which must be one of: 'SA', 'LA', 'SM', 'LM', 'EA', 'EM', 'IA', 'IM'. Got: {which}"
            )

        if Ms is not None:
            if not isinstance(Ms, np.ndarray):
                raise ValueError("Ms must be numpy array")
            # check shapes
            if Ms.shape != (Ms.shape[0], n, n):
                raise ValueError(
                    f"Ms must be of shape (p+1, n, n) [({Ms.shape[0]}, {n}, "
                    f"{n})], where p is the number of input features, got "
                    f"{Ms.shape}"
                )
            # check that Ms are Hermitian
            if not np.allclose(Ms, Ms.conj().transpose((0, 2, 1))):
                raise ValueError(
                    "Ms must be Hermitian (Ms[i] == Ms[i]^H for all i), got "
                    "non-Hermitian matrices"
                )

        self.n = n
        self.k = k
        self.p = (
            Ms.shape[0] - 1 if Ms is not None else None
        )  # number of input features
        self.which = which.lower()
        self.Ms = Ms  # matrices M0, M1, ..., Mp
        self.init_magnitude = init_magnitude

    def name(self) -> str:
        """
        Returns the name of the module
        """

        if self.k == 1 and self.which == "sa":
            return f"AffineEigenvaluePMM ({self.n}x{self.n}, ground state)"
        else:
            return f"AffineEigenvaluePMM ({self.n}x{self.n}, k={self.k}, which={self.which.upper()})"

    def is_ready(self) -> bool:
        return self.p is not None and self.Ms is not None

    def get_num_trainable_floats(self) -> Optional[int]:
        if not self.is_ready():
            return None

        # each matrix M is Hermitian, and so contains n * (n - 1) / 2 distinct
        # complex numbers and n distinct real numbers on the diagonal
        # the total number of trainable floats is then just n^2 per matrix
        return (self.p + 1) * self.n * self.n

    def _get_callable(self) -> Callable:
        """
        This method must return a jax-jittable and jax-gradable callable in the
        form of
        ```
        (
            params: Tuple[np.ndarray, ...],
            input_NF: np.ndarray[num_samples, num_features],
            training: bool,
            state: Tuple[np.ndarray, ...],
            rng: key<fry>
        ) -> (
                output_NF: np.ndarray[num_samples, num_output_features],
                new_state: Tuple[np.ndarray, ...]
            )
        ```
        That is, all hyperparameters are traced out and the callable depends
        explicitly only on a Tuple of parameter numpy arrays, the input array,
        the training flag, a state Tuple of numpy arrays, and a JAX rng key.

        The training flag will be traced out, so it doesn't need to be jittable
        """
        return lambda params, input_NF, training, state, rng: (
            affine_pmm_predict_func(
                params[0][0],  # A or M0
                params[0][1:],  # Bs or M1, M2, ..., Mp
                input_NF,  # cs
                self.k,  # k
                self.which,  # which
            ),
            state,  # state is not used in this module, return it unchanged
        )

    def compile(self, rng: Any, input_shape: Tuple[int, ...]) -> None:
        """
        Compile the module to be used with the given input shape.

        This method should initialize the module's parameters and state based
        on the input shape and random key.

        This is needed since Models are built before the input data is given,
        so before training or inference can be done, the module needs to be
        compiled and each Module passes its output shape to the next Module's
        compile method.

        The rng key is used to initialize random parameters, if needed.

        Parameters
        ----------
        rng : Any
            JAX random key.
        input_shape : Tuple[int, ...]
            Shape of the input data, e.g. (num_features,).
        """

        # input shape must be 1D
        if len(input_shape) != 1:
            raise ValueError(
                f"Input shape must be 1D, got {len(input_shape)}D shape: "
                f"{input_shape}"
            )

        # if the module is already ready, just verify the input shape
        if self.is_ready():
            if self.p != input_shape[0]:
                raise ValueError(
                    f"Input shape {input_shape} does not match the expected "
                    f"number of features {self.p}"
                )
            return

        # otherwise, initialize the matrices
        self.p = input_shape[0]  # number of input features

        subkey_real, subkey_imag = jax.random.split(rng, 2)
        self.Ms = self.init_magnitude * (
            jax.random.normal(
                subkey_real, (self.p + 1, self.n, self.n), dtype=np.complex64
            )
            + 1j
            * jax.random.normal(
                subkey_imag, (self.p + 1, self.n, self.n), dtype=np.complex64
            )
        )
        # ensure the matrices are Hermitian
        self.Ms = (self.Ms + self.Ms.conj().transpose((0, 2, 1))) / 2.0

    def get_output_shape(
        self, input_shape: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Get the output shape of the module given the input shape.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input data, e.g. (num_features,).

        Returns
        Tuple[int, ...]
            Shape of the output data, e.g. (num_output_features,).
        """
        return (self.k,)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the module.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the hyperparameters of the module.
        """
        return {
            "n": self.n,
            "k": self.k,
            "p": self.p,
            "which": self.which,
            "init_magnitude": self.init_magnitude,
        }

    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Set the hyperparameters of the module, using the default
        implementation. Just to input validation.

        Parameters
        ----------
        hyperparams : Dict[str, Any]
            Dictionary containing the hyperparameters to set.
        """
        if self.Ms is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(AffineEigenvaluePMM, self).set_hyperparameters(hyperparams)

    def get_params(self) -> Tuple[np.ndarray, ...]:
        """
        Get the current trainable parameters of the module. If the module has
        no trainable parameters, this method should return an empty tuple.

        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the module's parameters.
        """
        return (self.Ms,)

    def set_params(self, params: Tuple[np.ndarray, ...]) -> None:
        """
        Set the parameters of the module.

        Parameters
        ----------
        params : Tuple[np.ndarray, ...]
            Tuple of numpy arrays representing the new parameters.
        """
        if not isinstance(params, tuple) or not all(
            isinstance(p, np.ndarray) for p in params
        ):
            raise ValueError("params must be a tuple of numpy arrays")
        if len(params) != 1:
            raise ValueError(f"Expected 1 parameter array, got {len(params)}")
        if params[0].shape != (self.p + 1, self.n, self.n):
            raise ValueError(
                f"Parameter array must be of shape ({self.p + 1}, {self.n}, "
                f"{self.n}), got {params[0].shape}"
            )
        # check that the matrices are Hermitian
        if not np.allclose(params[0], params[0].conj().transpose((0, 2, 1))):
            raise ValueError("Parameter matrices must be Hermitian")

        self.Ms = params[0]
