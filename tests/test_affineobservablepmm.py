import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_affineobservablepmm():
    r"""
    Test the AffineObservablePMM module/model.
    """
    aopmm = pmm.modules.AffineObservablePMM

    key = jax.random.key(0)

    matrix_size = 5
    num_eig = 2
    num_observables = 1
    output_size = 1

    num_samples = 100

    # 2D Gaussian function, Z = exp(-((x-mux)^2/sx^2 + (y-muy)^2/sy^2 ))
    def f(x, y, mux, muy, sx, sy):
        return np.exp(-(((x - mux) ** 2) / sx**2 + ((y - muy) ** 2) / sy**2))

    mux, muy = 1.0, -1.0
    sx, sy = 0.5, 0.25

    xs = np.linspace(-3, 3, int(np.sqrt(num_samples)))
    ys = np.linspace(-3, 3, int(np.sqrt(num_samples)))
    xx, yy = np.meshgrid(xs, ys)

    X = np.vstack([xx.ravel(), yy.ravel()]).T  # [num_samples, 2]
    Y = f(xx.ravel(), yy.ravel(), mux, muy, sx, sy).reshape(
        -1, 1
    )  # [num_samples, 1]

    X_32 = X.astype(np.float32)
    Y_32 = Y.astype(np.float32)

    m = aopmm(
        matrix_size=matrix_size,
        num_eig=num_eig,
        output_size=output_size,
        num_secondaries=num_observables,
    )
    m.compile(key, pmm.tree_util.get_shapes(X, slice(1, None)))

    m.train(
        X_32,
        Y_32,
        lr=1e-2,
        epochs=1000,
        batch_size=25,
        batch_rng=key,
        verbose=False,
    )

    Y_pred = m.predict(X, dtype=np.float64)

    mse = np.mean((Y_pred - Y) ** 2)

    if mse >= 3e-6:
        raise ValueError(f"Test failed with MSE: {mse}")

    # test PyTree preservation
    X_tree = {"a": (X,)}
    Y_tree = {"a": (Y,)}

    X_tree_32 = pmm.tree_util.astype(X_tree, np.float32)
    Y_tree_32 = pmm.tree_util.astype(Y_tree, np.float32)

    m = aopmm(
        matrix_size=matrix_size,
        num_eig=num_eig,
        output_size=output_size,
        num_secondaries=num_observables,
    )
    m.compile(key, pmm.tree_util.get_shapes(X_tree, slice(1, None)))
    m.train(
        X_tree_32,
        Y_tree_32,
        lr=1e-2,
        epochs=1000,
        batch_size=25,
        batch_rng=key,
        verbose=False,
    )
    Y_tree_pred = m.predict(X_tree, dtype=np.float64)
    mse_tree = pmm.tree_util.mean(
        pmm.tree_util.abs_sqr(pmm.tree_util.sub(Y_tree_pred, Y_tree))
    )
    if mse_tree >= 3e-6:
        raise ValueError(f"PyTree test failed with MSE: {mse_tree}")
