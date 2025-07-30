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


class Abs(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Abs"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.abs(x, *self.args, **self.kwargs)


class Absolute(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Absolute"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.absolute(x, *self.args, **self.kwargs)


class ACos(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ACos"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.acos(x, *self.args, **self.kwargs)


class ACosh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ACosh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.acosh(x, *self.args, **self.kwargs)


class AMax(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "AMax"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.amax(x, *self.args, **self.kwargs)


class AMin(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "AMin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.amin(x, *self.args, **self.kwargs)


class Angle(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Angle"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.angle(x, *self.args, **self.kwargs)


class ArcCos(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcCos"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arccos(x, *self.args, **self.kwargs)


class ArcCosh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcCosh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arccosh(x, *self.args, **self.kwargs)


class ArcSin(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcSin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arcsin(x, *self.args, **self.kwargs)


class ArcSinh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcSinh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arcsinh(x, *self.args, **self.kwargs)


class ArcTan(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcTan"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctan(x, *self.args, **self.kwargs)


class ArcTan2(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcTan2"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctan2(x, *self.args, **self.kwargs)


class ArcTanh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ArcTanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.arctanh(x, *self.args, **self.kwargs)


class ASin(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ASin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.asin(x, *self.args, **self.kwargs)


class ASinh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ASinh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.asinh(x, *self.args, **self.kwargs)


class ATan(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ATan"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.atan(x, *self.args, **self.kwargs)


class ATanh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "ATanh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.atanh(x, *self.args, **self.kwargs)


class Cbrt(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Cbrt"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cbrt(x, *self.args, **self.kwargs)


class Ceil(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Ceil"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.ceil(x, *self.args, **self.kwargs)


class Clip(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Clip"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.clip(x, *self.args, **self.kwargs)


class Conj(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Conj"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.conj(x, *self.args, **self.kwargs)


class Conjugate(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Conjugate"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.conjugate(x, *self.args, **self.kwargs)


class Cos(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Cos"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cos(x, *self.args, **self.kwargs)


class Cosh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Cosh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.cosh(x, *self.args, **self.kwargs)


class Deg2Rad(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Deg2Rad"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.deg2rad(x, *self.args, **self.kwargs)


class Degrees(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Degrees"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.degrees(x, *self.args, **self.kwargs)


class Exp(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Exp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.exp(x, *self.args, **self.kwargs)


class Exp2(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Exp2"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.exp2(x, *self.args, **self.kwargs)


class Expm1(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Expm1"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.expm1(x, *self.args, **self.kwargs)


class FAbs(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FAbs"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.fabs(x, *self.args, **self.kwargs)


class Fix(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Fix"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.fix(x, *self.args, **self.kwargs)


class FloatPower(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FloatPower"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.float_power(x, *self.args, **self.kwargs)


class Floor(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Floor"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.floor(x, *self.args, **self.kwargs)


class FloorDivide(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FloorDivide"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.floor_divide(x, *self.args, **self.kwargs)


class FrExp(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "FrExp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.frexp(x, *self.args, **self.kwargs)


class I0(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "I0"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.i0(x, *self.args, **self.kwargs)


class Imag(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Imag"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.imag(x, *self.args, **self.kwargs)


class Invert(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Invert"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.invert(x, *self.args, **self.kwargs)


class LDExp(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "LDExp"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.ldexp(x, *self.args, **self.kwargs)


class Log(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log(x, *self.args, **self.kwargs)


class Log10(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log10"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log10(x, *self.args, **self.kwargs)


class Log1p(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log1p"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log1p(x, *self.args, **self.kwargs)


class Log2(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Log2"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.log2(x, *self.args, **self.kwargs)


class NaNToNum(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "NaNToNum"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nan_to_num(x, *self.args, **self.kwargs)


class NanToNum(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "NanToNum"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nan_to_num(x, *self.args, **self.kwargs)


class NextAfter(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "NextAfter"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.nextafter(x, *self.args, **self.kwargs)


class Packbits(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Packbits"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.packbits(x, *self.args, **self.kwargs)


class Piecewise(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Piecewise"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.piecewise(x, *self.args, **self.kwargs)


class Positive(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Positive"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.positive(x, *self.args, **self.kwargs)


class Pow(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Pow"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.pow(x, *self.args, **self.kwargs)


class Power(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Power"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.power(x, *self.args, **self.kwargs)


class Rad2Deg(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Rad2Deg"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.rad2deg(x, *self.args, **self.kwargs)


class Radians(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Radians"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.radians(x, *self.args, **self.kwargs)


class Real(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Real"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.real(x, *self.args, **self.kwargs)


class Reciprocal(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Reciprocal"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.reciprocal(x, *self.args, **self.kwargs)


class RInt(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "RInt"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.rint(x, *self.args, **self.kwargs)


class Round(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Round"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.round(x, *self.args, **self.kwargs)


class Sign(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sign"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sign(x, *self.args, **self.kwargs)


class Signbit(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Signbit"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.signbit(x, *self.args, **self.kwargs)


class Sin(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sin"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sin(x, *self.args, **self.kwargs)


class Sinc(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sinc"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sinc(x, *self.args, **self.kwargs)


class Sinh(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sinh"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sinh(x, *self.args, **self.kwargs)


class Sqrt(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Sqrt"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.sqrt(x, *self.args, **self.kwargs)


class Square(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Square"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.square(x, *self.args, **self.kwargs)


class Tan(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Tan"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.tan(x, *self.args, **self.kwargs)


class Trunc(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Trunc"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.trunc(x, *self.args, **self.kwargs)


class Unpackbits(ActivationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self) -> str:
        return "Unpackbits"

    def func(self, x: jax.numpy.ndarray) -> jax.numpy.ndarray:
        return jax.numpy.unpackbits(x, *self.args, **self.kwargs)


# This file is autogenerated. Do not edit manually.
