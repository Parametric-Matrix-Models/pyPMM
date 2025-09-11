from __future__ import annotations

import warnings
from typing import Any

import jax.numpy as np

from .affinehermitianmatrix import AffineHermitianMatrix
from .basemodule import BaseModule
from .bias import Bias
from .eigenvectors import Eigenvectors
from .multimodule import MultiModule
from .transitionamplitudesum import TransitionAmplitudeSum


class AffineObservablePMM(MultiModule):
    r"""
    ``AffineObservablePMM`` is a module that implements a general regression
    model via the affine observable
    Parametric Matrix Model (PMM) using four primitive modules combined in a
    MultiModule: a AffineHermitianMatrix module followed by an Eigenvectors
    module followed by a TransitionAmplitudeSum module followed optionally by a
    Bias module.

    The Affine Observable PMM (AOPMM) is described in [1]_ and is summarized as
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

    Then, take the leading :math:`r` eigenvectors (by default corresponding to
    the largest magnitude eigenvalues if there is no smoothing, or the smallest
    algebraic if there is smoothing) of :math:`M(x)` and compute the sum of the
    transition amplitudes of these eigenvectors with trainable Hermitian
    observable matrices (secondaries) :math:`D_{ql}` to form the output vector
    :math:`z` with :math:`q` components as

    .. math::

        z_k = \sum_{m=1}^l \left(
               \left[\sum_{i,j=1}^r |v_i^H D_{km} v_j|^2 \right]
                - \frac{r^2}{2} ||D_{km}||^2_2 \right)

    where :math:`||\cdot||_2` is the operator 2-norm (largest singular value)
    so for Hermitian :math:`D`, :math:`||D||_2` is the largest absolute
    eigenvalue.

    The :math:`-\frac{1}{2} ||D_{km}||^2_2` term centers the value of each term
    and can be disabled by setting the ``centered`` parameter to ``False``.

    Finally, an optional trainable bias term :math:`b_k` can be added to each
    component.

    .. warning::
        Even though the math shows that the centering term should be multiplied
        by :math:`r^2`, in practice this doesn't work well and instead setting
        the centering term to :math:`\frac{1}{2} ||D_{km}||^2_2` works much
        better. This non-:math:`r^2` scaling is used here.

    See Also
    --------
    AffineHermitianMatrix
        Module that constructs the affine Hermitian matrix :math:`M(x)` from
        trainable Hermitian matrices :math:`M_i` and input features.
    Eigenvectors
        Module that computes the eigenvectors of a matrix.
    TransitionAmplitudeSum
        Module that computes the sum of transition amplitudes of eigenvectors
        with trainable observable matrices.
    Bias
        Module that adds a trainable bias term to the output.
    MultiModule
        Module that combines multiple modules in sequence.
    LowRankAffineObservablePMM
        Low-rank version of this module.

    References
    ----------
    .. [1] Cook, P., Jammooa, D., Hjorth-Jensen, M. et al. Parametric matrix
            models. Nat Commun 16, 5929 (2025).
            https://doi.org/10.1038/s41467-025-61362-4
    """

    def __init__(
        self,
        matrix_size: int = None,
        num_eig: int = None,
        which: str = None,
        smoothing: float = None,
        affine_bias_matrix: bool = True,
        num_secondaries: int = 1,
        output_size: int = None,
        centered: bool = True,
        bias_term: bool = True,
        Ms: np.ndarray = None,
        Ds: np.ndarray = None,
        b: np.ndarray = None,
        init_magnitude: float = 0.01,
    ):
        r"""
        Initialize the module.

        Parameters
        ----------
            matrix_size
                Size of the trainable matrices, shorthand :math:`n`.
            num_eig
                Number of eigenvectors to use in the transition amplitude
                calculation, shorthand :math:`r`.
            which
                Which eigenvectors to use based on eigenvalue.
                Options are:
                - 'SA' for smallest algebraic (default with smoothing)
                - 'LA' for largest algebraic
                - 'SM' for smallest magnitude
                - 'LM' for largest magnitude (default without smoothing)
                - 'EA' for exterior algebraically
                - 'EM' for exterior by magnitude
                - 'IA' for interior algebraically
                - 'IM' for interior by magnitude
            smoothing
                Optional smoothing parameter for the affine matrix. Set to
                ``None``/``0.0`` to disable smoothing. Default is
                ``None``/``0.0`` (no smoothing).
            affine_bias_matrix
                If ``True``, include the bias term :math:`M_0` in the affine
                matrix. Default is ``True``.
            num_secondaries
                Number of secondary observable matrices :math:`D_{km}` per
                output component. Shorthand :math:`l`. Default is ``1``.
            output_size
                Size of the output vector, shorthand :math:`q`.
            centered
                If ``True``, include the centering term in the transition
                amplitude sum. Default is ``True``.
            bias_term
                If ``True``, include a trainable bias term :math:`b_k` in the
                output. Default is ``True``.
            Ms
                Optional array of shape
                ``(input_size+1, matrix_size, matrix_size)`` (if
                ``affine_bias_matrix`` is ``True``) or
                ``(input_size, matrix_size, matrix_size)`` (if
                ``affine_bias_matrix`` is ``False``), containing the
                :math:`M_i` Hermitian matrices. If not provided, the matrices
                will be initialized randomly when the module is compiled.
                Default is ``None`` (random initialization).
            Ds
                Optional array of shape
                ``(output_size, num_secondaries, matrix_size, matrix_size)``
                containing the :math:`D_{km}` Hermitian observable matrices. If
                not provided, the matrices will be initialized randomly when
                the module is compiled. Default is ``None`` (random
                initialization).
            b
                Optional array of shape ``(output_size,)`` containing the
                bias terms :math:`b_k`. If not provided, the bias terms will be
                randomly initialized when the module is compiled. Default is
                ``None`` (random initialization).

            init_magnitude
                Initial magnitude for the random initialization. Default is
                ``1e-2``.

        .. warning::
            When using smoothing, the ``which`` options involving magnitude
            should be avoided, as the smoothing only guarantees that
            eigenvalues near each other algebraically are smoothed, not across
            the spectrum.

        """

        # select default which based on smoothing
        if which is None:
            if smoothing in (None, 0.0):
                which = "LM"
            else:
                which = "SA"

        self.matrix_size = matrix_size
        self.num_eig = num_eig
        self.which = which
        self.smoothing = smoothing
        self.affine_bias_matrix = affine_bias_matrix
        self.num_secondaries = num_secondaries
        self.output_size = output_size
        self.centered = centered
        self.bias_term = bias_term
        self.Ms = Ms
        self.Ds = Ds
        self.b = b
        self.init_magnitude = init_magnitude

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
            AffineHermitianMatrix(
                matrix_size=matrix_size,
                smoothing=smoothing,
                Ms=Ms,
                init_magnitude=init_magnitude,
                bias_term=affine_bias_matrix,
                flatten=False,
            ),
            Eigenvectors(
                num_eig=num_eig,
                which=which,
            ),
            TransitionAmplitudeSum(
                num_observables=num_secondaries,
                output_size=output_size,
                centered=centered,
                Ds=Ds,
                init_magnitude=init_magnitude,
            ),
        )

        if bias_term:
            self.modules += (
                Bias(
                    bias=b,
                    init_magnitude=init_magnitude,
                    real=True,
                    scalar=False,
                    trainable=True,
                ),
            )

        super(AffineObservablePMM, self).__init__(*self.modules)

    def name(self) -> str:
        multistr = super(AffineObservablePMM, self).name()

        namestr = f"AffineObservablePMM as {multistr}"

        return namestr

    def get_hyperparameters(self) -> dict[str, Any]:
        data = {
            "matrix_size": self.matrix_size,
            "num_eig": self.num_eig,
            "which": self.which,
            "smoothing": self.smoothing,
            "init_magnitude": self.init_magnitude,
            "affine_bias_matrix": self.affine_bias_matrix,
            "num_secondaries": self.num_secondaries,
            "output_size": self.output_size,
            "centered": self.centered,
            "bias_term": self.bias_term,
        }

        return {
            **data,
            **super(AffineObservablePMM, self).get_hyperparameters(),
        }

    def set_hyperparameters(self, hyperparams: dict[str, Any]) -> None:
        self.matrix_size = hyperparams["matrix_size"]
        self.num_eig = hyperparams["num_eig"]
        self.which = hyperparams["which"]
        self.smoothing = hyperparams["smoothing"]
        self.init_magnitude = hyperparams["init_magnitude"]
        self.bias_term = hyperparams["bias_term"]
        self.affine_bias_matrix = hyperparams["affine_bias_matrix"]
        self.num_secondaries = hyperparams["num_secondaries"]
        self.output_size = hyperparams["output_size"]
        self.centered = hyperparams["centered"]
        self.parameter_counts = hyperparams["parameter_counts"]
        self.state_counts = hyperparams["state_counts"]
        self.input_shape = hyperparams["input_shape"]
        self.output_shape = hyperparams["output_shape"]

        self.modules = (
            AffineHermitianMatrix(
                matrix_size=self.matrix_size,
                smoothing=self.smoothing,
                Ms=self.Ms,
                init_magnitude=self.init_magnitude,
                bias_term=self.affine_bias_matrix,
                flatten=False,
            ),
            Eigenvectors(
                num_eig=self.num_eig,
                which=self.which,
            ),
            TransitionAmplitudeSum(
                num_observables=self.num_secondaries,
                output_size=self.output_size,
                centered=self.centered,
                Ds=self.Ds,
                init_magnitude=self.init_magnitude,
            ),
        )

        if self.bias_term:
            self.modules += (
                Bias(
                    bias=self.b,
                    init_magnitude=self.init_magnitude,
                    real=True,
                    scalar=False,
                    trainable=True,
                ),
            )

    def serialize(self) -> dict[str, Any]:
        # revert to BaseModule serialization
        return BaseModule.serialize(self)

    def deserialize(self, data: dict[str, Any]) -> None:
        # revert to BaseModule deserialization
        BaseModule.deserialize(self, data)
