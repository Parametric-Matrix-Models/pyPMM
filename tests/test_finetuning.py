import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_sequentialmodel_finetuning(tmp_path):
    r"""
    Test saving and loading a SequentialModel with training only part of the
    model between saving and loading.
    """

    key = jax.random.key(0)

    mag = 1.0

    batch_dim = 10
    input_data = mag * jax.random.normal(key, (batch_dim, 4)).astype(
        np.float32
    )

    def f(x):
        return np.sum(x**2, axis=-1, keepdims=True)

    y = f(input_data)

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

    model.train(input_data, Y=y, epochs=100, lr=1e-3)

    # store model parameters before saving
    params = model.get_params()

    save_path = tmp_path / "sequential_model.npz"
    model.save(save_path)

    loaded_model = pmm.SequentialModel.from_file(save_path)

    # freeze the first layer of the loaded model
    loaded_model.modules[0].trainable = False

    loaded_model.train(input_data, Y=y, epochs=100, lr=1e-3, resume=True)

    # check that the first layer parameters are unchanged
    close0 = jax.tree.map(
        lambda a, b: np.allclose(a, b, atol=1e-6),
        loaded_model.modules[0].get_params(),
        params[0],
    )
    assert jax.tree.all(close0)

    # check that the other layers parameters have changed
    close1 = jax.tree.map(
        lambda a, b: np.allclose(a, b, atol=1e-6),
        loaded_model.modules[1].get_params(),
        params[1],
    )
    close2 = jax.tree.map(
        lambda a, b: np.allclose(a, b, atol=1e-6),
        loaded_model.modules[2].get_params(),
        params[2],
    )
    assert not jax.tree.all(close1)
    assert not jax.tree.all(close2)

    # save the model again
    save_path_finetuned = tmp_path / "sequential_model_finetuned.npz"
    loaded_model.save(save_path_finetuned)

    # load the finetuned model
    finetuned_model = pmm.SequentialModel.from_file(save_path_finetuned)

    # make sure the first layer is still frozen
    assert not finetuned_model.modules[0].trainable

    # unfreeze the first layer and train again
    finetuned_model.modules[0].trainable = True

    finetuned_model.train(input_data, Y=y, epochs=100, lr=1e-3, resume=True)

    # check that now the first layer parameters have changed
    close0_after = jax.tree.map(
        lambda a, b: np.allclose(a, b, atol=1e-6),
        finetuned_model.modules[0].get_params(),
        params[0],
    )
    assert not jax.tree.all(close0_after)
