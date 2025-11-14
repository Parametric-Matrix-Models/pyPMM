from __future__ import annotations

import warnings
from typing import Any

import jax.numpy as np

from .basemodule import BaseModule
from .eigenvalues import Eigenvalues
from .lowrankaffinehermitianmatrix import LowRankAffineHermitianMatrix
from .multimodule import MultiModule


class LowRankAffineEigenvaluePMM(MultiModule):
    r"""
    ``LowRankAffineEigenvaluePMM`` is a module that implements the affine
    eigenvalue Parametric Matrix Model (PMM) using low-rank matrices via
    two primitive modules combined in a MultiModule: a
    LowRankAffineHermitianMatrix module followed by an Eigenvalues module.

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

    This module constructs the :math:`M_i` matrices as low-rank matrices, from
    outer products of trainable vectors.

    See Also
    --------
    LowRankAffineHermitianMatrix
        Module that constructs the affine Hermitian matrix :math:`M(x)` from
        low-rank matrices via outer products of trainable vectors.
    Eigenvalues
        Module that computes the eigenvalues of a matrix.
    AffineEigenvaluePMM
        Module that implements the affine eigenvalue PMM using full-rank
        matrices.
    MultiModule
        Module that combines multiple modules in sequence.

    References
    ----------
    .. [1] Cook, P., Jammooa, D., Hjorth-Jensen, M. et al. Parametric matrix
            models. Nat Commun 16, 5929 (2025).
            https://doi.org/10.1038/s41467-025-61362-4
    """

    def __init__(
        self,
        matrix_size: int = None,
        rank: int = None,
        num_eig: int = 1,
        which: str = "SA",
        smoothing: float = None,
        lambdas: np.ndarray = None,
        us: np.ndarray = None,
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
            rank
                Rank of the low-rank matrices, shorthand :math:`r`. Must be a
                positive integer less than or equal to ``matrix_size``.
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
            lambdas
                Optional array of shape `(input_size+1, rank)` (if
                ``bias_term`` is ``True``) or `(input_size, rank)` (if
                ``bias_term`` is ``False``), containing the `\lambda_k^i` real
                coefficients used to construct the low-rank Hermitian matrices.
                If not provided, the coefficients will be initialized randomly
                when the module is compiled.
            us
                Optional array of shape
                ``(input_size+1, rank, matrix_size)`` (if ``bias_term``
                is ``True``) or ``(input_size, rank, matrix_size)`` (if
                ``bias_term`` is ``False``), containing the :math:`u_k^i`
                complex vectors used to construct the low-rank Hermitian
                matrices. If not provided, the vectors will be
                initialized randomly when the module is compiled. Default is
                ``None`` (random initialization).
            init_magnitude
                Initial magnitude for the random matrices if ``us`` is not
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
        self.rank = rank
        self.num_eig = num_eig
        self.which = which
        self.smoothing = smoothing
        self.lambdas = lambdas
        self.us = us
        self.init_magnitude = init_magnitude
        self.bias_term = bias_term

        # raise a warning if smoothing is used with magnitude 'which'
        if smoothing not in (None, 0.0) and "m" in which.lower():
            warnings.warn(
                "Using smoothing with magnitude 'which' options may lead to "
                "unexpected behavior, as the smoothing only guarantees that "
                "eigenvalues near each other algebraically are smoothed, not "
                "across the spectrum.",
                UserWarning,
            )

        self.modules = (
            LowRankAffineHermitianMatrix(
                matrix_size=matrix_size,
                rank=rank,
                smoothing=smoothing,
                lambdas=lambdas,
                us=us,
                init_magnitude=init_magnitude,
                bias_term=bias_term,
                flatten=False,
            ),
            Eigenvalues(
                num_eig=num_eig,
                which=which,
            ),
        )

        super(LowRankAffineEigenvaluePMM, self).__init__(*self.modules)

    def name(self) -> str:
        multistr = super(LowRankAffineEigenvaluePMM, self).name()

        namestr = f"LowRankAffineEigenvaluePMM as {multistr}"

        return namestr

    def get_hyperparameters(self) -> dict[str, Any]:
        data = {
            "matrix_size": self.matrix_size,
            "rank": self.rank,
            "num_eig": self.num_eig,
            "which": self.which,
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
            "bias_term": self.bias_term,
        }

        return {
            **data,
            **super(LowRankAffineEigenvaluePMM, self).get_hyperparameters(),
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        self.matrix_size = hyperparams["matrix_size"]
        self.rank = hyperparams["rank"]
        self.num_eig = hyperparams["num_eig"]
        self.which = hyperparams["which"]
        self.smoothing = hyperparams["smoothing"]
        self.init_magnitude = hyperparams["init_magnitude"]
        self.bias_term = hyperparams["bias_term"]
        self.parameter_counts = hyperparams["parameter_counts"]
        self.state_counts = hyperparams["state_counts"]
        self.input_shape = hyperparams["input_shape"]
        self.output_shape = hyperparams["output_shape"]

        self.modules = (
            LowRankAffineHermitianMatrix(
                matrix_size=self.matrix_size,
                rank=self.rank,
                smoothing=self.smoothing,
                lambdas=self.lambdas,
                us=self.us,
                init_magnitude=self.init_magnitude,
                bias_term=self.bias_term,
                flatten=False,
            ),
            Eigenvalues(
                num_eig=self.num_eig,
                which=self.which,
            ),
        )

    def serialize(self) -> dict[str, Any]:
        # revert to BaseModule serialization
        return BaseModule.serialize(self)

    def deserialize(self, data: dict[str, Any]) -> None:
        # revert to BaseModule deserialization
        BaseModule.deserialize(self, data)
