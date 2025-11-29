from __future__ import annotations

import warnings

import jax
import jax.numpy as np

from ..sequentialmodel import SequentialModel
from ..tree_util import is_shape_leaf, is_single_leaf
from ..typing import (
    Any,
    DataShape,
    HyperParams,
    Tuple,
)
from .affinehermitianmatrix import AffineHermitianMatrix
from .basemodule import BaseModule
from .eigenvalues import Eigenvalues


class AffineEigenvaluePMM(SequentialModel):
    r"""
    ``AffineEigenvaluePMM`` is a module that implements the affine eigenvalue
    Parametric Matrix Model (PMM) using two primitive modules combined in a
    SequentialModel: an AffineHermitianMatrix module followed by an Eigenvalues
    module.

    The Affine Eigenvalue PMM (AEPMM) is described in [1]_ and is summarized as
    follows:

    Given input features :math:`x_1, \ldots, x_p`, construct the Hermitian
    matrix that is affine in these features as

    .. math::

        M(x) = M_0 + \sum_{i=1}^p x_i M_i

    where :math:`M_0, \ldots, M_p` are trainable Hermitian matrices. An
    optional smoothing term :math:`s C` parameterized by the smoothing
    hyperparameter :math:`s` can be added to smooth the eigenvalues and
    eigenvectors of :math:`M(x)`. The matrix :math:`C` is equal to the
    imaginary unit times the sum of all commutators of the :math:`M_i`.
    The requested eigenvalues of :math:`M(x)` are then computed and returned as
    the output of the module.

    See Also
    --------
    AffineHermitianMatrix
        Module that constructs the affine Hermitian matrix :math:`M(x)` from
        trainable Hermitian matrices :math:`M_i` and input features.
    Eigenvalues
        Module that computes the eigenvalues of a matrix.
    SequentialModel
        Module and Model that evaluates multiple modules in sequence.

    References
    ----------
    .. [1] Cook, P., Jammooa, D., Hjorth-Jensen, M. et al. Parametric matrix
            models. Nat Commun 16, 5929 (2025).
            https://doi.org/10.1038/s41467-025-61362-4
    """

    def __init__(
        self,
        matrix_size: int = None,
        num_eig: int = 1,
        which: str = "SA",
        smoothing: float = None,
        Ms: np.ndarray = None,
        init_magnitude: float = 0.01,
        bias_term: bool = True,
    ):
        r"""
        Initialize the ``AffineEigenvaluePMM`` module.

        By default this module is initialized to compute the smallest algebraic
        eigenvalue (ground state).

        Parameters
        ----------
            matrix_size
                Size of the PMM matrices, shorthand :math:`n`.
            num_eig
                Number of eigenvalues to compute, shorthand :math:`k`. Default
                is 1.
            which
                Which eigenvalues to compute. Default is "SA".
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
            smoothing
                Optional smoothing parameter. Set to ``0.0`` to disable
                smoothing. Default is ``None``/``0.0`` (no smoothing).
            Ms
                Optional array of shape
                ``(input_size+1, matrix_size, matrix_size)`` (if ``bias_term``
                is ``True``) or ``(input_size, matrix_size, matrix_size)`` (if
                ``bias_term`` is ``False``), containing the :math:`M_i`
                Hermitian matrices. If not provided, the matrices will be
                initialized randomly when the module is compiled. Default is
                ``None`` (random initialization).
            init_magnitude
                Initial magnitude for the random matrices if ``Ms`` is not
                provided. Default is ``1e-2``.
            bias_term
                If ``True``, include the bias term :math:`M_0` in the affine
                matrix. Default is ``True``.

        .. warning::
            When using smoothing, the ``which`` options involving magnitude
            should be avoided, as the smoothing only guarantees that
            eigenvalues near each other algebraically are smoothed, not across
            the spectrum.

        """

        self.matrix_size = matrix_size
        self.num_eig = num_eig
        self.which = which
        self.smoothing = smoothing
        self.Ms = Ms
        self.init_magnitude = init_magnitude
        self.bias_term = bias_term

        self.modules: Tuple[BaseModule] | None = None

        super().__init__()

    def compile(
        self,
        rng: Any | int | None,
        input_shape: DataShape,
        verbose: bool = False,
    ) -> None:
        valid, _ = is_single_leaf(input_shape, is_leaf=is_shape_leaf)
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf."
            )
        if self.matrix_size is None:
            raise ValueError("matrix_size must be specified before compiling.")
        # raise a warning if smoothing is used with magnitude 'which'
        if self.smoothing not in (None, 0.0) and "m" in self.which.lower():
            warnings.warn(
                "Using smoothing with magnitude 'which' options may lead to "
                "unexpected behavior, as the smoothing only guarantees that "
                "eigenvalues near each other algebraically are smoothed, not "
                "across the spectrum.",
                UserWarning,
            )

        if self.input_shape != input_shape or self.modules is None:
            # Create the AffineHermitianMatrix module
            affine_module = AffineHermitianMatrix(
                matrix_size=self.matrix_size,
                smoothing=self.smoothing,
                Ms=self.Ms,
                init_magnitude=self.init_magnitude,
                bias_term=self.bias_term,
            )

            # Create the Eigenvalues module
            eigen_module = Eigenvalues(
                num_eig=self.num_eig,
                which=self.which,
            )

            # Set the modules in the SequentialModel
            self.modules = (affine_module, eigen_module)

        # Call the parent compile method
        super().compile(rng, input_shape, verbose=verbose)

    def get_output_shape(self, input_shape: DataShape) -> DataShape:
        valid, _ = is_single_leaf(input_shape, is_leaf=is_shape_leaf)
        if not valid:
            raise ValueError(
                "Input shape must be a PyTree with a single leaf."
            )
        return jax.tree.map(
            lambda s: (self.num_eig,), input_shape, is_leaf=is_shape_leaf
        )

    def get_hyperparameters(self) -> HyperParams:
        return {
            "matrix_size": self.matrix_size,
            "num_eig": self.num_eig,
            "which": self.which,
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
            "bias_term": self.bias_term,
            **super().get_hyperparameters(),
        }

    def set_hyperparameters(self, hyperparams: HyperParams) -> None:
        self.matrix_size = hyperparams["matrix_size"]
        self.num_eig = hyperparams["num_eig"]
        self.which = hyperparams["which"]
        self.smoothing = hyperparams["smoothing"]
        self.init_magnitude = hyperparams["init_magnitude"]
        self.bias_term = hyperparams["bias_term"]
        super().set_hyperparameters(hyperparams)
