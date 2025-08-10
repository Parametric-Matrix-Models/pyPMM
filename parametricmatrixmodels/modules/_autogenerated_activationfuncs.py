from __future__ import annotations

import jax

from .activationbase import ActivationBase


class ReLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.relu``.

    See Also
    --------
    jax.nn.relu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ReLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.relu(x, *self.args, **self.kwargs)


class ReLU6(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.relu6``.

    See Also
    --------
    jax.nn.relu6 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ReLU6"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.relu6(x, *self.args, **self.kwargs)


class Sigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.sigmoid``.

    See Also
    --------
    jax.nn.sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sigmoid(x, *self.args, **self.kwargs)


class Softplus(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.softplus``.

    See Also
    --------
    jax.nn.softplus : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Softplus"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.softplus(x, *self.args, **self.kwargs)


class SparsePlus(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.sparse_plus``.

    See Also
    --------
    jax.nn.sparse_plus : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SparsePlus"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sparse_plus(x, *self.args, **self.kwargs)


class SparseSigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.sparse_sigmoid``.

    See Also
    --------
    jax.nn.sparse_sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SparseSigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sparse_sigmoid(x, *self.args, **self.kwargs)


class SoftSign(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.soft_sign``.

    See Also
    --------
    jax.nn.soft_sign : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SoftSign"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.soft_sign(x, *self.args, **self.kwargs)


class SiLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.silu``.

    See Also
    --------
    jax.nn.silu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SiLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.silu(x, *self.args, **self.kwargs)


class Swish(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.swish``.

    See Also
    --------
    jax.nn.swish : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Swish"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.swish(x, *self.args, **self.kwargs)


class LogSigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.log_sigmoid``.

    See Also
    --------
    jax.nn.log_sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LogSigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.log_sigmoid(x, *self.args, **self.kwargs)


class LeakyReLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.leaky_relu``.

    See Also
    --------
    jax.nn.leaky_relu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LeakyReLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.leaky_relu(x, *self.args, **self.kwargs)


class HardSigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_sigmoid``.

    See Also
    --------
    jax.nn.hard_sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardSigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_sigmoid(x, *self.args, **self.kwargs)


class HardSiLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_silu``.

    See Also
    --------
    jax.nn.hard_silu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardSiLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_silu(x, *self.args, **self.kwargs)


class HardSwish(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_swish``.

    See Also
    --------
    jax.nn.hard_swish : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardSwish"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_swish(x, *self.args, **self.kwargs)


class HardTanh(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_tanh``.

    See Also
    --------
    jax.nn.hard_tanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardTanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_tanh(x, *self.args, **self.kwargs)


class ELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.elu``.

    See Also
    --------
    jax.nn.elu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.elu(x, *self.args, **self.kwargs)


class CELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.celu``.

    See Also
    --------
    jax.nn.celu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "CELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.celu(x, *self.args, **self.kwargs)


class SELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.selu``.

    See Also
    --------
    jax.nn.selu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.selu(x, *self.args, **self.kwargs)


class GELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.gelu``.

    See Also
    --------
    jax.nn.gelu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "GELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.gelu(x, *self.args, **self.kwargs)


class GLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.glu``.

    See Also
    --------
    jax.nn.glu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "GLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.glu(x, *self.args, **self.kwargs)


class SquarePlus(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.squareplus``.

    See Also
    --------
    jax.nn.squareplus : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SquarePlus"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.squareplus(x, *self.args, **self.kwargs)


class Mish(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.mish``.

    See Also
    --------
    jax.nn.mish : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Mish"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.mish(x, *self.args, **self.kwargs)


class Identity(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.identity``.

    See Also
    --------
    jax.nn.identity : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Identity"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.identity(x, *self.args, **self.kwargs)


class Softmax(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.softmax``.

    See Also
    --------
    jax.nn.softmax : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Softmax"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.softmax(x, *self.args, **self.kwargs)


class LogSoftmax(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.log_softmax``.

    See Also
    --------
    jax.nn.log_softmax : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LogSoftmax"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.log_softmax(x, *self.args, **self.kwargs)


class LogSumExp(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.logsumexp``.

    See Also
    --------
    jax.nn.logsumexp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LogSumExp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.logsumexp(x, *self.args, **self.kwargs)


class Standardize(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.standardize``.

    See Also
    --------
    jax.nn.standardize : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Standardize"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.standardize(x, *self.args, **self.kwargs)


class OneHot(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.one_hot``.

    See Also
    --------
    jax.nn.one_hot : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "OneHot"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.one_hot(x, *self.args, **self.kwargs)


class Abs(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.abs``.

    See Also
    --------
    jax.numpy.abs : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Abs"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.abs(x, *self.args, **self.kwargs)


class Absolute(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.absolute``.

    See Also
    --------
    jax.numpy.absolute : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Absolute"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.absolute(x, *self.args, **self.kwargs)


class ACos(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.acos``.

    See Also
    --------
    jax.numpy.acos : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ACos"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.acos(x, *self.args, **self.kwargs)


class ACosh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.acosh``.

    See Also
    --------
    jax.numpy.acosh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ACosh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.acosh(x, *self.args, **self.kwargs)


class AMax(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.amax``.

    See Also
    --------
    jax.numpy.amax : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "AMax"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.amax(x, *self.args, **self.kwargs)


class AMin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.amin``.

    See Also
    --------
    jax.numpy.amin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "AMin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.amin(x, *self.args, **self.kwargs)


class Angle(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.angle``.

    See Also
    --------
    jax.numpy.angle : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Angle"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.angle(x, *self.args, **self.kwargs)


class ArcCos(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arccos``.

    See Also
    --------
    jax.numpy.arccos : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcCos"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arccos(x, *self.args, **self.kwargs)


class ArcCosh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arccosh``.

    See Also
    --------
    jax.numpy.arccosh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcCosh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arccosh(x, *self.args, **self.kwargs)


class ArcSin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arcsin``.

    See Also
    --------
    jax.numpy.arcsin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcSin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arcsin(x, *self.args, **self.kwargs)


class ArcSinh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arcsinh``.

    See Also
    --------
    jax.numpy.arcsinh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcSinh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arcsinh(x, *self.args, **self.kwargs)


class ArcTan(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arctan``.

    See Also
    --------
    jax.numpy.arctan : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcTan"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctan(x, *self.args, **self.kwargs)


class ArcTan2(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arctan2``.

    See Also
    --------
    jax.numpy.arctan2 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcTan2"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctan2(x, *self.args, **self.kwargs)


class ArcTanh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arctanh``.

    See Also
    --------
    jax.numpy.arctanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcTanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctanh(x, *self.args, **self.kwargs)


class ASin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.asin``.

    See Also
    --------
    jax.numpy.asin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ASin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.asin(x, *self.args, **self.kwargs)


class ASinh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.asinh``.

    See Also
    --------
    jax.numpy.asinh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ASinh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.asinh(x, *self.args, **self.kwargs)


class ATan(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.atan``.

    See Also
    --------
    jax.numpy.atan : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ATan"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.atan(x, *self.args, **self.kwargs)


class ATanh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.atanh``.

    See Also
    --------
    jax.numpy.atanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ATanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.atanh(x, *self.args, **self.kwargs)


class Cbrt(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.cbrt``.

    See Also
    --------
    jax.numpy.cbrt : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Cbrt"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cbrt(x, *self.args, **self.kwargs)


class Ceil(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.ceil``.

    See Also
    --------
    jax.numpy.ceil : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Ceil"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.ceil(x, *self.args, **self.kwargs)


class Clip(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.clip``.

    See Also
    --------
    jax.numpy.clip : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Clip"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.clip(x, *self.args, **self.kwargs)


class Conj(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.conj``.

    See Also
    --------
    jax.numpy.conj : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Conj"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.conj(x, *self.args, **self.kwargs)


class Conjugate(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.conjugate``.

    See Also
    --------
    jax.numpy.conjugate : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Conjugate"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.conjugate(x, *self.args, **self.kwargs)


class Cos(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.cos``.

    See Also
    --------
    jax.numpy.cos : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Cos"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cos(x, *self.args, **self.kwargs)


class Cosh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.cosh``.

    See Also
    --------
    jax.numpy.cosh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Cosh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cosh(x, *self.args, **self.kwargs)


class Deg2Rad(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.deg2rad``.

    See Also
    --------
    jax.numpy.deg2rad : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Deg2Rad"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.deg2rad(x, *self.args, **self.kwargs)


class Degrees(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.degrees``.

    See Also
    --------
    jax.numpy.degrees : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Degrees"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.degrees(x, *self.args, **self.kwargs)


class Exp(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.exp``.

    See Also
    --------
    jax.numpy.exp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Exp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.exp(x, *self.args, **self.kwargs)


class Exp2(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.exp2``.

    See Also
    --------
    jax.numpy.exp2 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Exp2"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.exp2(x, *self.args, **self.kwargs)


class Expm1(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.expm1``.

    See Also
    --------
    jax.numpy.expm1 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Expm1"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.expm1(x, *self.args, **self.kwargs)


class FAbs(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.fabs``.

    See Also
    --------
    jax.numpy.fabs : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FAbs"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.fabs(x, *self.args, **self.kwargs)


class Fix(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.fix``.

    See Also
    --------
    jax.numpy.fix : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Fix"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.fix(x, *self.args, **self.kwargs)


class FloatPower(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.float_power``.

    See Also
    --------
    jax.numpy.float_power : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FloatPower"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.float_power(x, *self.args, **self.kwargs)


class Floor(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.floor``.

    See Also
    --------
    jax.numpy.floor : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Floor"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.floor(x, *self.args, **self.kwargs)


class FloorDivide(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.floor_divide``.

    See Also
    --------
    jax.numpy.floor_divide : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FloorDivide"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.floor_divide(x, *self.args, **self.kwargs)


class FrExp(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.frexp``.

    See Also
    --------
    jax.numpy.frexp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FrExp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.frexp(x, *self.args, **self.kwargs)


class I0(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.i0``.

    See Also
    --------
    jax.numpy.i0 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "I0"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.i0(x, *self.args, **self.kwargs)


class Imag(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.imag``.

    See Also
    --------
    jax.numpy.imag : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Imag"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.imag(x, *self.args, **self.kwargs)


class Invert(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.invert``.

    See Also
    --------
    jax.numpy.invert : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Invert"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.invert(x, *self.args, **self.kwargs)


class LDExp(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.ldexp``.

    See Also
    --------
    jax.numpy.ldexp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LDExp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.ldexp(x, *self.args, **self.kwargs)


class Log(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log``.

    See Also
    --------
    jax.numpy.log : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log(x, *self.args, **self.kwargs)


class Log10(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log10``.

    See Also
    --------
    jax.numpy.log10 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log10"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log10(x, *self.args, **self.kwargs)


class Log1p(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log1p``.

    See Also
    --------
    jax.numpy.log1p : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log1p"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log1p(x, *self.args, **self.kwargs)


class Log2(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log2``.

    See Also
    --------
    jax.numpy.log2 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log2"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log2(x, *self.args, **self.kwargs)


class NaNToNum(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.nan_to_num``.

    See Also
    --------
    jax.numpy.nan_to_num : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "NaNToNum"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nan_to_num(x, *self.args, **self.kwargs)


class NanToNum(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.nan_to_num``.

    See Also
    --------
    jax.numpy.nan_to_num : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "NanToNum"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nan_to_num(x, *self.args, **self.kwargs)


class NextAfter(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.nextafter``.

    See Also
    --------
    jax.numpy.nextafter : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "NextAfter"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nextafter(x, *self.args, **self.kwargs)


class Packbits(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.packbits``.

    See Also
    --------
    jax.numpy.packbits : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Packbits"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.packbits(x, *self.args, **self.kwargs)


class Piecewise(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.piecewise``.

    See Also
    --------
    jax.numpy.piecewise : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Piecewise"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.piecewise(x, *self.args, **self.kwargs)


class Positive(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.positive``.

    See Also
    --------
    jax.numpy.positive : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Positive"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.positive(x, *self.args, **self.kwargs)


class Pow(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.pow``.

    See Also
    --------
    jax.numpy.pow : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Pow"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.pow(x, *self.args, **self.kwargs)


class Power(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.power``.

    See Also
    --------
    jax.numpy.power : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Power"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.power(x, *self.args, **self.kwargs)


class Rad2Deg(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.rad2deg``.

    See Also
    --------
    jax.numpy.rad2deg : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Rad2Deg"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.rad2deg(x, *self.args, **self.kwargs)


class Radians(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.radians``.

    See Also
    --------
    jax.numpy.radians : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Radians"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.radians(x, *self.args, **self.kwargs)


class Real(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.real``.

    See Also
    --------
    jax.numpy.real : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Real"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.real(x, *self.args, **self.kwargs)


class Reciprocal(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.reciprocal``.

    See Also
    --------
    jax.numpy.reciprocal : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Reciprocal"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.reciprocal(x, *self.args, **self.kwargs)


class RInt(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.rint``.

    See Also
    --------
    jax.numpy.rint : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "RInt"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.rint(x, *self.args, **self.kwargs)


class Round(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.round``.

    See Also
    --------
    jax.numpy.round : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Round"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.round(x, *self.args, **self.kwargs)


class Sign(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sign``.

    See Also
    --------
    jax.numpy.sign : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sign"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sign(x, *self.args, **self.kwargs)


class Signbit(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.signbit``.

    See Also
    --------
    jax.numpy.signbit : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Signbit"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.signbit(x, *self.args, **self.kwargs)


class Sin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sin``.

    See Also
    --------
    jax.numpy.sin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sin(x, *self.args, **self.kwargs)


class Sinc(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sinc``.

    See Also
    --------
    jax.numpy.sinc : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sinc"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sinc(x, *self.args, **self.kwargs)


class Sinh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sinh``.

    See Also
    --------
    jax.numpy.sinh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sinh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sinh(x, *self.args, **self.kwargs)


class Sqrt(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sqrt``.

    See Also
    --------
    jax.numpy.sqrt : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sqrt"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sqrt(x, *self.args, **self.kwargs)


class Square(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.square``.

    See Also
    --------
    jax.numpy.square : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Square"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.square(x, *self.args, **self.kwargs)


class Tan(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.tan``.

    See Also
    --------
    jax.numpy.tan : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Tan"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.tan(x, *self.args, **self.kwargs)


class Tanh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.tanh``.

    See Also
    --------
    jax.numpy.tanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Tanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.tanh(x, *self.args, **self.kwargs)


class Trunc(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.trunc``.

    See Also
    --------
    jax.numpy.trunc : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Trunc"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.trunc(x, *self.args, **self.kwargs)


class Unpackbits(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.unpackbits``.

    See Also
    --------
    jax.numpy.unpackbits : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Unpackbits"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.unpackbits(x, *self.args, **self.kwargs)


# This file is autogenerated. Do not edit manually.
