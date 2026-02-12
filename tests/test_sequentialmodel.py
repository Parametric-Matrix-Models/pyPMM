import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_sequentialmodel_trivial():
    r"""
    Test the SequentialModel class with trivial examples.
    """

    modules = [pmm.modules.Constant(1.0)]

    model = pmm.SequentialModel(modules)

    # test with array inputs

    batch_dim = 10
    input_data = (
        np.arange(0, batch_dim * 12)
        .reshape((batch_dim, 3, 4))
        .astype(np.float64)
    )

    model.compile(None, input_data.shape[1:])

    output = model(input_data)

    assert output.shape == (batch_dim,)
    assert np.allclose(output, 1.0)

    model.append(pmm.modules.Bias(0.5, scalar=True))

    model.compile(None, input_data.shape[1:])

    output = model(input_data)
    assert output.shape == (batch_dim,)
    assert np.allclose(output, 1.5)

    # test with PyTree inputs, and PyTree modules
    modules = {
        "first": pmm.modules.Bias(0.5, scalar=True),
        "seconds": (
            pmm.modules.Reshape([(2, 6), (8,)]),
            pmm.modules.ReLU(),
        ),
    }

    model = pmm.SequentialModel(modules)

    input_data = [
        np.arange(batch_dim * 12)
        .reshape((batch_dim, 3, 4))
        .astype(np.float64),
        np.arange(batch_dim * 8)
        .reshape((batch_dim, 2, 2, 2))
        .astype(np.float64),
    ]
    model.compile(None, [data.shape[1:] for data in input_data])

    output = model(input_data)

    assert isinstance(output, list)
    assert output[0].shape == (batch_dim, 2, 6)
    assert output[1].shape == (batch_dim, 8)

    expected_0 = np.maximum(
        np.arange(batch_dim * 12).reshape((batch_dim, 2, 6)) + 0.5, 0
    )
    expected_1 = np.maximum(
        np.arange(batch_dim * 8).reshape((batch_dim, 8)) + 0.5, 0
    )

    assert np.allclose(output[0], expected_0)
    assert np.allclose(output[1], expected_1)


def test_sequentialmodel_linear():
    r"""
    Test the SequentialModel class with linear layers
    """

    key = jax.random.key(0)

    batch_dim = 10
    input_data = jax.random.normal(key, (batch_dim, 4))

    modules = [
        pmm.modules.LinearNN(
            out_features=8, bias=True, activation=pmm.modules.ReLU()
        ),
        pmm.modules.LinearNN(
            out_features=8, bias=True, activation=pmm.modules.ReLU()
        ),
        pmm.modules.LinearNN(out_features=1, bias=True),
    ]

    model = pmm.SequentialModel(modules)

    model.compile(key, input_data.shape[1:])
    output = model(input_data)

    assert output.shape == (batch_dim, 1)

    # manually compute the output
    params = model.get_params()
    x = input_data
    for i in range(len(modules)):
        if len(params[i]) == 3:
            # no activation
            _, W, b = params[i]
            x = x @ W.T + b
        else:
            # relu activation
            _, W, b, _ = params[i]
            x = np.maximum(x @ W.T + b, 0)
    expected_output = x
    assert output.shape == expected_output.shape
    assert np.allclose(output, expected_output)


def test_sequential_save_load(tmp_path):
    r"""
    Test saving and loading a SequentialModel.
    """

    key = jax.random.key(0)

    mag = 1.0

    batch_dim = 10
    input_data = mag * jax.random.normal(key, (batch_dim, 4))

    modules = [
        pmm.modules.LinearNN(
            out_features=8,
            bias=True,
            activation=pmm.modules.ReLU(),
            init_magnitude=mag,
        ),
        pmm.modules.LinearNN(
            out_features=8,
            bias=True,
            activation=pmm.modules.ReLU(),
            init_magnitude=mag,
        ),
        pmm.modules.LinearNN(out_features=1, bias=True, init_magnitude=mag),
    ]

    model = pmm.SequentialModel(modules)

    model.compile(key, input_data.shape[1:])
    output = model(input_data)

    save_path = tmp_path / "sequential_model.npz"
    model.save(save_path)

    loaded_model = pmm.SequentialModel.from_file(save_path)

    loaded_output = loaded_model(input_data)

    assert np.allclose(output, loaded_output)
