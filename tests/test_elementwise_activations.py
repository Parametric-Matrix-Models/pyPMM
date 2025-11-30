import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

# name: (module class, jax function, *args, **kwargs)
funcs = {
    "ReLU": (pmm.modules.ReLU, jax.nn.relu, (), {}),
    "ReLU6": (pmm.modules.ReLU6, jax.nn.relu6, (), {}),
    "Sigmoid": (pmm.modules.Sigmoid, jax.nn.sigmoid, (), {}),
    "Softplus": (pmm.modules.Softplus, jax.nn.softplus, (), {}),
    "SparsePlus": (pmm.modules.SparsePlus, jax.nn.sparse_plus, (), {}),
    "SparseSigmoid": (
        pmm.modules.SparseSigmoid,
        jax.nn.sparse_sigmoid,
        (),
        {},
    ),
    "SoftSign": (pmm.modules.SoftSign, jax.nn.soft_sign, (), {}),
    "SiLU": (pmm.modules.SiLU, jax.nn.silu, (), {}),
    "Swish": (pmm.modules.Swish, jax.nn.swish, (), {}),
    "LogSigmoid": (pmm.modules.LogSigmoid, jax.nn.log_sigmoid, (), {}),
    "LeakyReLU": (pmm.modules.LeakyReLU, jax.nn.leaky_relu, (), {}),
    "HardSigmoid": (pmm.modules.HardSigmoid, jax.nn.hard_sigmoid, (), {}),
    "HardSiLU": (pmm.modules.HardSiLU, jax.nn.hard_silu, (), {}),
    "HardSwish": (pmm.modules.HardSwish, jax.nn.hard_swish, (), {}),
    "HardTanh": (pmm.modules.HardTanh, jax.nn.hard_tanh, (), {}),
    "ELU": (pmm.modules.ELU, jax.nn.elu, (), {}),
    "CELU": (pmm.modules.CELU, jax.nn.celu, (), {}),
    "SELU": (pmm.modules.SELU, jax.nn.selu, (), {}),
    "GELU": (pmm.modules.GELU, jax.nn.gelu, (), {}),
    "GLU": (pmm.modules.GLU, jax.nn.glu, (), {}),
    "SquarePlus": (pmm.modules.SquarePlus, jax.nn.squareplus, (), {}),
    "Mish": (pmm.modules.Mish, jax.nn.mish, (), {}),
    "Identity": (pmm.modules.Identity, jax.nn.identity, (), {}),
    "Softmax": (pmm.modules.Softmax, jax.nn.softmax, (), {}),
    "LogSoftmax": (pmm.modules.LogSoftmax, jax.nn.log_softmax, (), {}),
    "LogSumExp": (pmm.modules.LogSumExp, jax.nn.logsumexp, (), {"axis": -1}),
    "Standardize": (pmm.modules.Standardize, jax.nn.standardize, (), {}),
    # "OneHot": (pmm.modules.OneHot, jax.nn.one_hot, (10,), {}), # needs
    # integer inputs
    "Abs": (pmm.modules.Abs, jax.numpy.abs, (), {}),
    "Absolute": (pmm.modules.Absolute, jax.numpy.absolute, (), {}),
    "ACos": (pmm.modules.ACos, jax.numpy.acos, (), {}),
    "ACosh": (pmm.modules.ACosh, jax.numpy.acosh, (), {}),
    "AMax": (pmm.modules.AMax, jax.numpy.amax, (), {"axis": -1}),
    "AMin": (pmm.modules.AMin, jax.numpy.amin, (), {"axis": -1}),
    "Angle": (pmm.modules.Angle, jax.numpy.angle, (), {}),
    "ArcCos": (pmm.modules.ArcCos, jax.numpy.arccos, (), {}),
    "ArcCosh": (pmm.modules.ArcCosh, jax.numpy.arccosh, (), {}),
    "ArcSin": (pmm.modules.ArcSin, jax.numpy.arcsin, (), {}),
    "ArcSinh": (pmm.modules.ArcSinh, jax.numpy.arcsinh, (), {}),
    "ArcTan": (pmm.modules.ArcTan, jax.numpy.arctan, (), {}),
    "ArcTan2": (pmm.modules.ArcTan2, jax.numpy.arctan2, (1.0,), {}),
    "ArcTanh": (pmm.modules.ArcTanh, jax.numpy.arctanh, (), {}),
    "ASin": (pmm.modules.ASin, jax.numpy.asin, (), {}),
    "ASinh": (pmm.modules.ASinh, jax.numpy.asinh, (), {}),
    "ATan": (pmm.modules.ATan, jax.numpy.atan, (), {}),
    "ATanh": (pmm.modules.ATanh, jax.numpy.atanh, (), {}),
    "Cbrt": (pmm.modules.Cbrt, jax.numpy.cbrt, (), {}),
    "Ceil": (pmm.modules.Ceil, jax.numpy.ceil, (), {}),
    "Clip": (pmm.modules.Clip, jax.numpy.clip, (), {}),
    "Conj": (pmm.modules.Conj, jax.numpy.conj, (), {}),
    "Conjugate": (pmm.modules.Conjugate, jax.numpy.conjugate, (), {}),
    "Cos": (pmm.modules.Cos, jax.numpy.cos, (), {}),
    "Cosh": (pmm.modules.Cosh, jax.numpy.cosh, (), {}),
    "Deg2Rad": (pmm.modules.Deg2Rad, jax.numpy.deg2rad, (), {}),
    "Degrees": (pmm.modules.Degrees, jax.numpy.degrees, (), {}),
    "Exp": (pmm.modules.Exp, jax.numpy.exp, (), {}),
    "Exp2": (pmm.modules.Exp2, jax.numpy.exp2, (), {}),
    "Expm1": (pmm.modules.Expm1, jax.numpy.expm1, (), {}),
    "FAbs": (pmm.modules.FAbs, jax.numpy.fabs, (), {}),
    "Fix": (pmm.modules.Fix, jax.numpy.fix, (), {}),
    "FloatPower": (pmm.modules.FloatPower, jax.numpy.float_power, (1.0,), {}),
    "Floor": (pmm.modules.Floor, jax.numpy.floor, (), {}),
    "FloorDivide": (
        pmm.modules.FloorDivide,
        jax.numpy.floor_divide,
        (2.0,),
        {},
    ),
    # "FrExp": (pmm.modules.FrExp, jax.numpy.frexp, (), {}), # returns tuple,
    # so can't test directly automatically
    "I0": (pmm.modules.I0, jax.numpy.i0, (), {}),
    "Imag": (pmm.modules.Imag, jax.numpy.imag, (), {}),
    # "Invert": (pmm.modules.Invert, jax.numpy.invert, (), {}), # only for
    # integer and bool types
    # "LDExp": (pmm.modules.LDExp, jax.numpy.ldexp, (1.0,), {}), # does not
    # support float32
    "Log": (pmm.modules.Log, jax.numpy.log, (), {}),
    "Log10": (pmm.modules.Log10, jax.numpy.log10, (), {}),
    "Log1p": (pmm.modules.Log1p, jax.numpy.log1p, (), {}),
    "Log2": (pmm.modules.Log2, jax.numpy.log2, (), {}),
    "NaNToNum": (pmm.modules.NaNToNum, jax.numpy.nan_to_num, (), {}),
    "NanToNum": (pmm.modules.NanToNum, jax.numpy.nan_to_num, (), {}),
    "NextAfter": (pmm.modules.NextAfter, jax.numpy.nextafter, (1.0,), {}),
    # "Packbits": (pmm.modules.Packbits, jax.numpy.packbits, (), {}), # only
    # for integer and boolean types
    # "Piecewise": (pmm.modules.Piecewise, jax.numpy.piecewise, (), {}), #
    # needs complicated args that match shapes
    "Positive": (pmm.modules.Positive, jax.numpy.positive, (), {}),
    "Pow": (pmm.modules.Pow, jax.numpy.pow, (1.0,), {}),
    "Power": (pmm.modules.Power, jax.numpy.power, (1.0,), {}),
    "Rad2Deg": (pmm.modules.Rad2Deg, jax.numpy.rad2deg, (), {}),
    "Radians": (pmm.modules.Radians, jax.numpy.radians, (), {}),
    "Real": (pmm.modules.Real, jax.numpy.real, (), {}),
    "Reciprocal": (pmm.modules.Reciprocal, jax.numpy.reciprocal, (), {}),
    "RInt": (pmm.modules.RInt, jax.numpy.rint, (), {}),
    "Round": (pmm.modules.Round, jax.numpy.round, (), {}),
    "Sign": (pmm.modules.Sign, jax.numpy.sign, (), {}),
    # "Signbit": (pmm.modules.Signbit, jax.numpy.signbit, (), {}), # returns
    # boolean array, which may need special handling
    "Sin": (pmm.modules.Sin, jax.numpy.sin, (), {}),
    "Sinc": (pmm.modules.Sinc, jax.numpy.sinc, (), {}),
    "Sinh": (pmm.modules.Sinh, jax.numpy.sinh, (), {}),
    "Sqrt": (pmm.modules.Sqrt, jax.numpy.sqrt, (), {}),
    "Square": (pmm.modules.Square, jax.numpy.square, (), {}),
    "Tan": (pmm.modules.Tan, jax.numpy.tan, (), {}),
    "Tanh": (pmm.modules.Tanh, jax.numpy.tanh, (), {}),
    "Trunc": (pmm.modules.Trunc, jax.numpy.trunc, (), {}),
    # "Unpackbits": (pmm.modules.Unpackbits, jax.numpy.unpackbits, (), {}),
    # only for unsigned byte inputs
}


def _test_name(module: pmm.modules.BaseModule) -> None:
    _ = module.name  # just check that it runs without error


def _test_array_accuracy(
    module: pmm.modules.BaseModule,
    arraydata: pmm.typing.ArrayData,
    func: callable,
    args: tuple,
    kwargs: dict,
) -> None:
    r"""
    Helper function to test module accuracy on array data.
    """

    out, _ = module(arraydata)
    expected = func(arraydata, *args, **kwargs)
    assert np.allclose(out, expected, equal_nan=True)


def _test_pytree_accuracy(
    module: pmm.modules.BaseModule,
    data: pmm.typing.Data,
    func: callable,
    args: tuple,
    kwargs: dict,
) -> None:
    r"""
    Helper function to test module accuracy on pytree data.
    """
    out, _ = module(data)
    expected = jax.tree.map(lambda d: func(d, *args, **kwargs), data)
    assert all(
        [
            np.allclose(o, e, equal_nan=True)
            for o, e in zip(jax.tree.leaves(out), jax.tree.leaves(expected))
        ]
    )


def test_activations():
    r"""
    Test all element-wise activation functions.
    """
    # check with array and pytree inputs
    input_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    input_pytree = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]),
    )

    for name, (ModuleClass, jax_func, args, kwargs) in funcs.items():
        module = ModuleClass(*args, **kwargs)

        # Test name property
        _test_name(module)

        # Test with array input
        _test_array_accuracy(module, input_array, jax_func, args, kwargs)

        # Test with pytree input
        _test_pytree_accuracy(module, input_pytree, jax_func, args, kwargs)
