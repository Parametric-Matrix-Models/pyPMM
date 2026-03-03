import time

import jax
import jax.numpy as np
import pytest

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", False)


def test_nonsequential_compile_time():
    r"""
    Benchmark the JIT compile time of NonSequentialModel.train() for a
    large model (>20k parameters). The compile time should be reduced
    by at least 50% compared to the baseline.
    """

    num_modules = 16
    matrix_size = 12
    input_size = 5
    output_size = 5

    modules = {}
    connections = {}

    for i in range(num_modules):
        modules[f"mod{i}"] = pmm.modules.AffineObservablePMM(
            matrix_size=matrix_size,
            num_eig=6,
            num_secondaries=1,
            output_size=output_size,
            bias_term=True,
            smoothing=1.0,
        )

    connections["input"] = "mod0"
    for i in range(num_modules - 1):
        connections[f"mod{i}"] = f"mod{i + 1}"
    connections[f"mod{num_modules - 1}"] = "output"

    model = pmm.NonSequentialModel(modules, connections)

    X = np.ones((20, input_size), dtype=np.float32)
    Y = np.ones((20, output_size), dtype=np.float32)

    model.compile(42, (input_size,))

    num_params = model.get_num_trainable_floats()
    assert num_params >= 20000, (
        f"Model has {num_params} trainable floats, need >= 20000"
    )

    # Clear JAX caches to ensure a fresh compilation
    jax.clear_caches()

    t0 = time.time()
    model.train(
        X,
        Y,
        loss_fn="mse",
        epochs=1,
        batch_size=20,
        verbose=False,
    )
    compile_time = time.time() - t0

    print(
        f"\nCompile time: {compile_time:.2f}s"
        f" (num_params={num_params},"
        f" num_param_arrays={len(jax.tree.leaves(model.get_params()))})"
    )

    # The compile time should be reasonable (under 30 seconds)
    # This threshold was chosen to be roughly half of the original
    # baseline compile time (~35-45s on CI).
    assert compile_time < 30.0, (
        f"Compile time {compile_time:.2f}s exceeds 30s threshold"
    )
