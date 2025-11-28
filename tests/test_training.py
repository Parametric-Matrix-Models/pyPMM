import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_sequentialmodel_train_array():
    r"""
    Test training the SequentialModel with array inputs and outputs
    """

    key = jax.random.key(0)

    num_samples = 100
    num_features = 2
    X = np.linspace(-1, 1, num_samples * num_features).reshape(
        num_samples, num_features
    )
    Y = np.sin(np.pi * X).sum(axis=1, keepdims=True)

    modules = [
        pmm.modules.LinearNN(
            out_features=8,
            bias=True,
            activation=pmm.modules.ReLU(),
            init_magnitude=0.1,
        ),
        pmm.modules.LinearNN(
            out_features=8,
            bias=True,
            activation=pmm.modules.ReLU(),
            init_magnitude=0.1,
        ),
        pmm.modules.LinearNN(
            out_features=1, bias=True, activation=None, init_magnitude=0.1
        ),
    ]

    model = pmm.SequentialModel(modules)
    model.compile(key, X.shape[1:])
    model.astype(np.float32)

    X_32 = X.astype(np.float32)
    Y_32 = Y.astype(np.float32)

    model.train(
        X_32,
        Y_32,
        lr=5e-3,
        epochs=1000,
        batch_size=25,
        batch_rng=key,
        verbose=False,
    )

    # predict with 64-bit precision
    Y_pred = model.predict(X)

    mse = np.mean((Y - Y_pred) ** 2)

    if mse >= 2e-3:
        raise AssertionError(f"MSE {mse:.4E} is greater than expected 2E-3")


def test_sequentialmodel_train_pytree():
    r"""
    Test training the SequentialModel with pytee inputs and outputs

    In this case just a single matmul to vector model, i.e. just a linear model

    The input is a PyTree (list) of arrays and the output is a single array.
    """

    key = jax.random.key(0)

    num_samples = 100
    num_features = 2
    num_leaves = 3
    keys = jax.random.split(key, num_leaves + 1)
    X = [
        jax.random.uniform(
            keys[i],
            minval=-1,
            maxval=1,
            shape=(num_samples, num_features, num_features),
        )
        for i in range(num_leaves - 1)
    ]
    X.append(
        jax.random.uniform(
            keys[~1], minval=-1, maxval=1, shape=(num_samples, num_features)
        )
    )
    Y = np.einsum(
        "ij,njk,nkl,nl->ni",
        jax.random.uniform(
            keys[~0], minval=-1, maxval=1, shape=(num_features, num_features)
        ),
        *X,
    )

    modules = [
        pmm.modules.MatMul(output_shape=num_features, trainable=True),
    ]

    model = pmm.SequentialModel(modules)
    model.compile(key, jax.tree.map(lambda x: x.shape[1:], X))
    model.astype(np.float32)

    X_32 = pmm.tree_util.astype(X, np.float32)
    Y_32 = pmm.tree_util.astype(Y, np.float32)

    model.train(
        X_32,
        Y_32,
        lr=1e-2,
        epochs=1000,
        batch_size=50,
        batch_rng=key,
        verbose=False,
    )

    # predict with 64-bit precision
    Y_pred = model.predict(X)

    mse = pmm.tree_util.mean(
        pmm.tree_util.abs_sqr(pmm.tree_util.sub(Y, Y_pred))
    )

    if mse >= 1e-8:
        raise AssertionError(f"MSE {mse:.4E} is greater than expected 1E-8")
