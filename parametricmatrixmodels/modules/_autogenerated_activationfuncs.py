import jax
from beartype import beartype
from jaxtyping import jaxtyped

from .activationbase import ActivationBase


@jaxtyped(typechecker=beartype)
class ReLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.relu``.

    See Also
    --------
    jax.nn.relu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.relu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ReLU6(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.relu6``.

    See Also
    --------
    jax.nn.relu6 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.relu6(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Sigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.sigmoid``.

    See Also
    --------
    jax.nn.sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sigmoid(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Softplus(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.softplus``.

    See Also
    --------
    jax.nn.softplus : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.softplus(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class SparsePlus(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.sparse_plus``.

    See Also
    --------
    jax.nn.sparse_plus : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sparse_plus(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class SparseSigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.sparse_sigmoid``.

    See Also
    --------
    jax.nn.sparse_sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sparse_sigmoid(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class SoftSign(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.soft_sign``.

    See Also
    --------
    jax.nn.soft_sign : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.soft_sign(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class SiLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.silu``.

    See Also
    --------
    jax.nn.silu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.silu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Swish(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.swish``.

    See Also
    --------
    jax.nn.swish : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.swish(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class LogSigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.log_sigmoid``.

    See Also
    --------
    jax.nn.log_sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.log_sigmoid(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class LeakyReLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.leaky_relu``.

    See Also
    --------
    jax.nn.leaky_relu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.leaky_relu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class HardSigmoid(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_sigmoid``.

    See Also
    --------
    jax.nn.hard_sigmoid : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_sigmoid(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class HardSiLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_silu``.

    See Also
    --------
    jax.nn.hard_silu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_silu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class HardSwish(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_swish``.

    See Also
    --------
    jax.nn.hard_swish : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_swish(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class HardTanh(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.hard_tanh``.

    See Also
    --------
    jax.nn.hard_tanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_tanh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.elu``.

    See Also
    --------
    jax.nn.elu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.elu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class CELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.celu``.

    See Also
    --------
    jax.nn.celu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.celu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class SELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.selu``.

    See Also
    --------
    jax.nn.selu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.selu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class GELU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.gelu``.

    See Also
    --------
    jax.nn.gelu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.gelu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class GLU(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.glu``.

    See Also
    --------
    jax.nn.glu : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.glu(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class SquarePlus(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.squareplus``.

    See Also
    --------
    jax.nn.squareplus : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.squareplus(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Mish(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.mish``.

    See Also
    --------
    jax.nn.mish : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.mish(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Identity(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.identity``.

    See Also
    --------
    jax.nn.identity : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.identity(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Softmax(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.softmax``.

    See Also
    --------
    jax.nn.softmax : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.softmax(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class LogSoftmax(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.log_softmax``.

    See Also
    --------
    jax.nn.log_softmax : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.log_softmax(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class LogSumExp(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.logsumexp``.

    See Also
    --------
    jax.nn.logsumexp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.logsumexp(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Standardize(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.standardize``.

    See Also
    --------
    jax.nn.standardize : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.standardize(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class OneHot(ActivationBase):
    """
    Elementwise activation function for ``jax.nn.one_hot``.

    See Also
    --------
    jax.nn.one_hot : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.one_hot(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Abs(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.abs``.

    See Also
    --------
    jax.numpy.abs : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.abs(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Absolute(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.absolute``.

    See Also
    --------
    jax.numpy.absolute : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.absolute(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ACos(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.acos``.

    See Also
    --------
    jax.numpy.acos : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.acos(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ACosh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.acosh``.

    See Also
    --------
    jax.numpy.acosh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.acosh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class AMax(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.amax``.

    See Also
    --------
    jax.numpy.amax : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.amax(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class AMin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.amin``.

    See Also
    --------
    jax.numpy.amin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.amin(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Angle(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.angle``.

    See Also
    --------
    jax.numpy.angle : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.angle(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ArcCos(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arccos``.

    See Also
    --------
    jax.numpy.arccos : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arccos(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ArcCosh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arccosh``.

    See Also
    --------
    jax.numpy.arccosh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arccosh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ArcSin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arcsin``.

    See Also
    --------
    jax.numpy.arcsin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arcsin(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ArcSinh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arcsinh``.

    See Also
    --------
    jax.numpy.arcsinh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arcsinh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ArcTan(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arctan``.

    See Also
    --------
    jax.numpy.arctan : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctan(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ArcTan2(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arctan2``.

    See Also
    --------
    jax.numpy.arctan2 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctan2(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ArcTanh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.arctanh``.

    See Also
    --------
    jax.numpy.arctanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctanh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ASin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.asin``.

    See Also
    --------
    jax.numpy.asin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.asin(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ASinh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.asinh``.

    See Also
    --------
    jax.numpy.asinh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.asinh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ATan(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.atan``.

    See Also
    --------
    jax.numpy.atan : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.atan(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class ATanh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.atanh``.

    See Also
    --------
    jax.numpy.atanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.atanh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Cbrt(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.cbrt``.

    See Also
    --------
    jax.numpy.cbrt : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cbrt(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Ceil(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.ceil``.

    See Also
    --------
    jax.numpy.ceil : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.ceil(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Clip(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.clip``.

    See Also
    --------
    jax.numpy.clip : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.clip(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Conj(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.conj``.

    See Also
    --------
    jax.numpy.conj : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.conj(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Conjugate(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.conjugate``.

    See Also
    --------
    jax.numpy.conjugate : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.conjugate(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Cos(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.cos``.

    See Also
    --------
    jax.numpy.cos : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cos(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Cosh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.cosh``.

    See Also
    --------
    jax.numpy.cosh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cosh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Deg2Rad(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.deg2rad``.

    See Also
    --------
    jax.numpy.deg2rad : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.deg2rad(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Degrees(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.degrees``.

    See Also
    --------
    jax.numpy.degrees : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.degrees(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Exp(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.exp``.

    See Also
    --------
    jax.numpy.exp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.exp(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Exp2(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.exp2``.

    See Also
    --------
    jax.numpy.exp2 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.exp2(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Expm1(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.expm1``.

    See Also
    --------
    jax.numpy.expm1 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.expm1(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class FAbs(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.fabs``.

    See Also
    --------
    jax.numpy.fabs : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.fabs(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Fix(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.fix``.

    See Also
    --------
    jax.numpy.fix : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.fix(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class FloatPower(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.float_power``.

    See Also
    --------
    jax.numpy.float_power : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.float_power(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Floor(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.floor``.

    See Also
    --------
    jax.numpy.floor : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.floor(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class FloorDivide(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.floor_divide``.

    See Also
    --------
    jax.numpy.floor_divide : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.floor_divide(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class FrExp(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.frexp``.

    See Also
    --------
    jax.numpy.frexp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.frexp(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class I0(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.i0``.

    See Also
    --------
    jax.numpy.i0 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.i0(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Imag(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.imag``.

    See Also
    --------
    jax.numpy.imag : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.imag(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Invert(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.invert``.

    See Also
    --------
    jax.numpy.invert : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.invert(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class LDExp(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.ldexp``.

    See Also
    --------
    jax.numpy.ldexp : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.ldexp(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Log(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log``.

    See Also
    --------
    jax.numpy.log : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Log10(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log10``.

    See Also
    --------
    jax.numpy.log10 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log10(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Log1p(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log1p``.

    See Also
    --------
    jax.numpy.log1p : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log1p(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Log2(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.log2``.

    See Also
    --------
    jax.numpy.log2 : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log2(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class NaNToNum(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.nan_to_num``.

    See Also
    --------
    jax.numpy.nan_to_num : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nan_to_num(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class NanToNum(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.nan_to_num``.

    See Also
    --------
    jax.numpy.nan_to_num : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nan_to_num(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class NextAfter(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.nextafter``.

    See Also
    --------
    jax.numpy.nextafter : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nextafter(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Packbits(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.packbits``.

    See Also
    --------
    jax.numpy.packbits : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.packbits(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Piecewise(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.piecewise``.

    See Also
    --------
    jax.numpy.piecewise : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.piecewise(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Positive(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.positive``.

    See Also
    --------
    jax.numpy.positive : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.positive(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Pow(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.pow``.

    See Also
    --------
    jax.numpy.pow : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.pow(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Power(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.power``.

    See Also
    --------
    jax.numpy.power : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.power(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Rad2Deg(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.rad2deg``.

    See Also
    --------
    jax.numpy.rad2deg : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.rad2deg(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Radians(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.radians``.

    See Also
    --------
    jax.numpy.radians : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.radians(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Real(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.real``.

    See Also
    --------
    jax.numpy.real : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.real(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Reciprocal(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.reciprocal``.

    See Also
    --------
    jax.numpy.reciprocal : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.reciprocal(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class RInt(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.rint``.

    See Also
    --------
    jax.numpy.rint : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.rint(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Round(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.round``.

    See Also
    --------
    jax.numpy.round : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.round(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Sign(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sign``.

    See Also
    --------
    jax.numpy.sign : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sign(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Signbit(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.signbit``.

    See Also
    --------
    jax.numpy.signbit : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.signbit(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Sin(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sin``.

    See Also
    --------
    jax.numpy.sin : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sin(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Sinc(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sinc``.

    See Also
    --------
    jax.numpy.sinc : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sinc(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Sinh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sinh``.

    See Also
    --------
    jax.numpy.sinh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sinh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Sqrt(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.sqrt``.

    See Also
    --------
    jax.numpy.sqrt : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sqrt(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Square(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.square``.

    See Also
    --------
    jax.numpy.square : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.square(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Tan(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.tan``.

    See Also
    --------
    jax.numpy.tan : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.tan(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Tanh(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.tanh``.

    See Also
    --------
    jax.numpy.tanh : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.tanh(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Trunc(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.trunc``.

    See Also
    --------
    jax.numpy.trunc : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.trunc(x, *self.args, **self.kwargs)


@jaxtyped(typechecker=beartype)
class Unpackbits(ActivationBase):
    """
    Elementwise activation function for ``jax.numpy.unpackbits``.

    See Also
    --------
    jax.numpy.unpackbits : The function used for the elementwise activation.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.unpackbits(x, *self.args, **self.kwargs)


# This file is autogenerated. Do not edit manually.
