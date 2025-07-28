import jax

from .activationbase import ActivationBase


class ReLU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ReLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.relu(x, *self.args, **self.kwargs)


class ReLU6(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ReLU6"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.relu6(x, *self.args, **self.kwargs)


class Sigmoid(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sigmoid(x, *self.args, **self.kwargs)


class Softplus(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Softplus"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.softplus(x, *self.args, **self.kwargs)


class SparsePlus(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SparsePlus"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sparse_plus(x, *self.args, **self.kwargs)


class SparseSigmoid(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SparseSigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.sparse_sigmoid(x, *self.args, **self.kwargs)


class SoftSign(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SoftSign"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.soft_sign(x, *self.args, **self.kwargs)


class SiLU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SiLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.silu(x, *self.args, **self.kwargs)


class Swish(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Swish"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.swish(x, *self.args, **self.kwargs)


class LogSigmoid(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LogSigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.log_sigmoid(x, *self.args, **self.kwargs)


class LeakyReLU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LeakyReLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.leaky_relu(x, *self.args, **self.kwargs)


class HardSigmoid(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardSigmoid"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_sigmoid(x, *self.args, **self.kwargs)


class HardSiLU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardSiLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_silu(x, *self.args, **self.kwargs)


class HardSwish(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardSwish"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_swish(x, *self.args, **self.kwargs)


class HardTanh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "HardTanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.hard_tanh(x, *self.args, **self.kwargs)


class ELU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.elu(x, *self.args, **self.kwargs)


class CELU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "CELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.celu(x, *self.args, **self.kwargs)


class SELU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.selu(x, *self.args, **self.kwargs)


class GELU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "GELU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.gelu(x, *self.args, **self.kwargs)


class GLU(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "GLU"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.glu(x, *self.args, **self.kwargs)


class SquarePlus(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "SquarePlus"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.squareplus(x, *self.args, **self.kwargs)


class Mish(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Mish"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.mish(x, *self.args, **self.kwargs)


class Identity(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Identity"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.identity(x, *self.args, **self.kwargs)


class Softmax(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Softmax"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.softmax(x, *self.args, **self.kwargs)


class LogSoftmax(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LogSoftmax"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.log_softmax(x, *self.args, **self.kwargs)


class Tanh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Tanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.tanh(x, *self.args, **self.kwargs)


class LogSumExp(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LogSumExp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.logsumexp(x, *self.args, **self.kwargs)


class Standardize(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Standardize"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.standardize(x, *self.args, **self.kwargs)


class OneHot(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "OneHot"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.nn.one_hot(x, *self.args, **self.kwargs)


# This file is autogenerated. Do not edit manually.
