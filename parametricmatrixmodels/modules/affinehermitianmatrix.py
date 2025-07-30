from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as np

from ._regression_backing_funcs import exact_smoothing_matrix
from .basemodule import BaseModule


class AffineHermitianMatrix(BaseModule):

    def __init__(
        self,
        matrix_size: int = None,
        smoothing: Optional[float] = None,
        Ms: Optional[np.ndarray] = None,
        init_magnitude: float = 1e-2,
        flatten: bool = False,
    ) -> None:
        """
        M(x) = M0 + x1 * M1 + ... + xp * Mp + smoothing * C

        Parameters
        ----------
            matrix_size : int
                Size of the PMM matrices (square), shorthand `n`.
            smoothing : Optional[float], optional
                Smoothing parameter, set to 0.0 to disable smoothing.
                Defaults to None/0.0.
            Ms : Optional[np.ndarray], optional
                Optional array of matrices M0, M1, ..., Mp that define the
                parametric affine matrix. Each M must be Hermitian. If not
                provided, the matrices will be randomly initialized when the
                module is compiled.
            init_magnitude : float, optional
                Initial magnitude of the random matrices, used when
                initializing the module. Defaults to 1e-2.
            flatten : bool, optional
                If True, the output will be flattened to a 1D array. Useful
                when combining with SubsetModule or other modules in order to
                avoid ragged arrays.

        """

        # input validation
        if matrix_size is not None and (
            matrix_size <= 0 or not isinstance(matrix_size, int)
        ):
            raise ValueError("matrix_size must be a positive integer")
        if Ms is not None:
            if not isinstance(Ms, np.ndarray):
                raise ValueError("Ms must be a numpy array")
            matrix_size = matrix_size or Ms.shape[1]
            if Ms.shape != (Ms.shape[0], matrix_size, matrix_size):
                raise ValueError(
                    "Ms must be a 3D array of shape (input_size+1,"
                    f" matrix_size, matrix_size) [({Ms.shape[0]},"
                    f" {matrix_size}, {matrix_size})], got {Ms.shape}"
                )
            # ensure Ms are Hermitian
            if not np.allclose(Ms, Ms.conj().transpose((0, 2, 1))):
                raise ValueError("Ms must be Hermitian matrices")

        self.matrix_size = matrix_size
        self.smoothing = smoothing if smoothing is not None else 0.0
        self.Ms = Ms  # matrices M0, M1, ..., Mp
        self.init_magnitude = init_magnitude
        self.flatten = flatten

    def name(self) -> str:
        """
        Returns the name of the module
        """

        return (
            f"AffineHermitianMatrix({self.matrix_size}x{self.matrix_size},"
            f" smoothing={self.smoothing}"
            f"{', FLATTENED' if self.flatten else ''})"
        )

    def is_ready(self) -> bool:
        return self.Ms is not None

    def get_num_trainable_floats(self) -> Optional[int]:
        if not self.is_ready():
            return None

        # each matrix M is Hermitian, and so contains n * (n - 1) / 2 distinct
        # complex numbers and n distinct real numbers on the diagonal
        # the total number of trainable floats is then just n^2 per matrix
        # so Ms contributes (p + 1) * n^2 floats

        return self.Ms.size

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

        def affine_hermitian_matrix(
            params: Tuple[np.ndarray, ...],
            input_NF: np.ndarray,
            training: bool,
            state: Tuple[np.ndarray, ...],
            rng: Any,
        ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:

            Ms = params[0]

            # enforce Hermitian matrices
            Ms = (Ms + Ms.conj().transpose((0, 2, 1))) / 2.0

            M = Ms[0][None, :, :] + np.einsum(
                "ni,ijk->njk", input_NF.astype(Ms.dtype), Ms[1:]
            )

            if self.smoothing != 0.0:
                M += (
                    self.smoothing
                    * exact_smoothing_matrix(Ms[0], Ms[1:])[None, :, :]
                )

            if self.flatten:
                # preserve batch dimension
                return (M.reshape(M.shape[0], -1), state)
            else:
                return (M, state)

        return affine_hermitian_matrix

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
            if self.Ms.shape[0] != input_shape[0] + 1:
                raise ValueError(
                    f"Input shape {input_shape} does not match the expected "
                    f"number of features {self.Ms.shape[0] - 1} "
                )
            return

        rng_Mreal, rng_Mimag = jax.random.split(rng, 2)

        self.Ms = self.init_magnitude * (
            jax.random.normal(
                rng_Mreal,
                (input_shape[0] + 1, self.matrix_size, self.matrix_size),
                dtype=np.complex64,
            )
            + 1j
            * jax.random.normal(
                rng_Mimag,
                (input_shape[0] + 1, self.matrix_size, self.matrix_size),
                dtype=np.complex64,
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
        if self.flatten:
            return (self.matrix_size**2,)
        else:
            return (self.matrix_size, self.matrix_size)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters of the module.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the hyperparameters of the module.
        """
        return {
            "matrix_size": self.matrix_size,
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
            "flatten": self.flatten,
        }

    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Set the hyperparameters of the module using the default implementation,
        just do input validation.

        Parameters
        ----------
        hyperparams : Dict[str, Any]
            Dictionary containing the hyperparameters to set.
        """
        if self.Ms is not None:
            raise ValueError(
                "Cannot set hyperparameters after the module has parameters"
            )

        super(AffineHermitianMatrix, self).set_hyperparameters(hyperparams)

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
            raise ValueError(f"Expected 1 parameter arrays, got {len(params)}")

        Ms = params[0]

        expected_shape = (
            Ms.shape[0] if self.Ms is None else self.Ms.shape[0],
            self.matrix_size,
            self.matrix_size,
        )

        if Ms.shape != expected_shape:
            raise ValueError(
                "Ms must be a 3D array of shape (input_size+1, matrix_size,"
                f" matrix_size) [{expected_shape}], got {Ms.shape}"
            )
        # ensure Ms are Hermitian
        if not np.allclose(Ms, Ms.conj().transpose((0, 2, 1))):
            raise ValueError("Ms must be Hermitian matrices")

        self.Ms = Ms
