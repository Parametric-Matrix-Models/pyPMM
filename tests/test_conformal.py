import jax
import jax.numpy as np

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_conformal_sequentialarray():
    r"""
    Test ConformalizedModel on a SequentialModel with array values
    """

    key = jax.random.key(0)

    trkey, ckey, tekey, ntrkey, nckey, pkey, key = jax.random.split(key, 7)

    alphas = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]  # miscoverage levels

    def f(x):
        return x**3 - 2 * x * x + 1.0

    xlim = (-0.5, 2.0)

    # generate lightly noised abs(x) data

    n_train = 20
    x_train = jax.random.uniform(
        trkey, (n_train,), minval=xlim[0], maxval=xlim[1]
    )
    y_train = f(x_train) + 0.05 * jax.random.normal(ntrkey, (n_train,))

    n_calib = 500
    x_calib = jax.random.uniform(
        ckey, (n_calib,), minval=xlim[0], maxval=xlim[1]
    )
    y_calib = f(x_calib) + 0.05 * jax.random.normal(nckey, (n_calib,))

    # add feature axis
    x_train = x_train[:, None]
    x_calib = x_calib[:, None]
    y_train = y_train[:, None]
    y_calib = y_calib[:, None]

    # make a NN model
    im = 1e-1

    modules = [
        pmm.modules.LinearNN(
            out_features=8,
            bias=True,
            activation=pmm.modules.Softplus(),
            init_magnitude=im,
        ),
        pmm.modules.LinearNN(
            out_features=8,
            bias=True,
            activation=pmm.modules.Softplus(),
            init_magnitude=im,
        ),
        pmm.modules.LinearNN(
            out_features=1,
            bias=True,
            activation=None,
            init_magnitude=im,
        ),
    ]

    model = pmm.SequentialModel(modules)
    model.compile(key, (1,))

    # train
    model.train(
        pmm.tree_util.astype(x_train, np.float32),
        Y=pmm.tree_util.astype(y_train, np.float32),
        epochs=500,
        batch_size=32,
        lr=1e-2,
        batch_rng=key,
        verbose=True,
    )

    # make conformal model
    cmodel = pmm.conformal.ConformalizedModel(model)

    # calibrate
    cmodel.calibrate(x_calib, y_calib, X_train=x_train)

    # make test data
    N_test = 5000
    x_test = np.linspace(xlim[0], xlim[1], N_test)[:, None]
    y_test = f(x_test)

    for alpha in alphas:
        # get prediction intervals
        y_pred, (y_lower, y_upper) = cmodel(x_test, alpha=alpha)

        # confirm that at least (1-alpha) fraction of the test points are
        # covered
        covered = np.logical_and(y_test >= y_lower, y_test <= y_upper)
        coverage = np.mean(covered)

        if coverage < 1 - alpha - 0.01:
            raise ValueError(f"Coverage {coverage} is less than {1 - alpha}.")
        if coverage > 1 - alpha + 0.10:
            print(
                f"Warning: Coverage {coverage} is much greater than"
                f" {1 - alpha}."
            )


def test_conformal_nonsequentialpytree():
    r"""
    Test ConformalizedModel on a non-SequentialModel with pytree values
    """

    alphas = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]  # miscoverage levels

    # fit to slightly noised 2D Runge's function

    key = jax.random.key(0)

    dkey, nkey, pkey, key = jax.random.split(key, 4)

    # generate lightly noised Runge's function data
    N = 300
    x = jax.random.uniform(dkey, (N, 2), minval=-1.0, maxval=1.0)

    def runge_function(x):
        r = np.sqrt(np.sum(x**2, axis=-1))
        return 10.0 / (1.0 + 25 * r**2)

    y = runge_function(x) + 0.05 * jax.random.normal(nkey, (N,))
    y = y[:, None]

    # convert to PyTrees
    X = {"x": x[:, 0:1], "y": x[:, 1:2]}
    Y = {"value": y}

    # split into train and calibration sets
    n_train = 100
    X_shuffle = jax.tree.map(lambda x: jax.random.permutation(pkey, x), X)
    Y_shuffle = jax.tree.map(lambda y: jax.random.permutation(pkey, y), Y)
    X_train = jax.tree.map(lambda x: x[:n_train], X_shuffle)
    Y_train = jax.tree.map(lambda y: y[:n_train], Y_shuffle)
    X_calib = jax.tree.map(lambda x: x[n_train:], X_shuffle)
    Y_calib = jax.tree.map(lambda y: y[n_train:], Y_shuffle)

    # make non-sequential model
    # the input is first flattened and concatenated, then
    # features are fed into an observable pmm to generate new features,
    # then the original features and the new features concatenated and
    # flattened and fed into a linear nn
    # finally the output is re-keyed into a pytree with the same structure as Y
    modules = {
        "pmm": pmm.modules.AffineObservablePMM(
            matrix_size=5,
            num_eig=2,
            smoothing=1.0,
            num_secondaries=1,
            output_size=5,
        ),
        "nn": pmm.modules.LinearNN(out_features=1, bias=True, activation=None),
        "concat1": pmm.modules.ConcatenateLeaves(),
        "concat2": pmm.modules.ConcatenateLeaves(),
        "key": pmm.modules.TreeKey({"value": ""}),
    }
    connections = {
        "input": "concat1",
        "concat1": ["pmm", "concat2"],
        "pmm": "concat2",
        "concat2": "nn",
        "nn": "key",
        "key": "output",
    }
    model = pmm.NonSequentialModel(modules, connections)
    model.compile(
        key, pmm.tree_util.get_shapes(X, axis=slice(1, None)), verbose=True
    )
    # train
    model.train(
        pmm.tree_util.astype(X_train, np.float32),
        Y=pmm.tree_util.astype(Y_train, np.float32),
        epochs=200,
        batch_size=20,
        lr=1e-3,
        batch_rng=key,
        verbose=True,
    )

    # make conformal model
    cmodel = pmm.conformal.ConformalizedModel(model)
    # calibrate
    cmodel.calibrate(X_calib, Y_calib, X_train=X_train)
    # make test data
    N_test = 400
    x1 = np.linspace(-1.2, 1.2, int(np.sqrt(N_test)))
    x2 = np.linspace(-1.2, 1.2, int(np.sqrt(N_test)))
    x1g, x2g = np.meshgrid(x1, x2)
    x_test = np.stack([x1g.ravel(), x2g.ravel()], axis=-1)
    y_test = runge_function(x_test)[:, None]
    # convert to PyTree
    X_test = {"x": x_test[:, 0:1], "y": x_test[:, 1:2]}
    Y_test = {"value": y_test}

    for alpha in alphas:
        # get prediction intervals
        Y_pred, (Y_lower, Y_upper) = cmodel(X_test, alpha=alpha)
        # confirm that at least (1-alpha) fraction of the test points are
        # covered
        covered = np.logical_and(
            Y_test["value"] >= Y_lower["value"],
            Y_test["value"] <= Y_upper["value"],
        )
        coverage = np.mean(covered)

        if coverage < 1 - alpha - 0.01:
            raise ValueError(f"Coverage {coverage} is less than {1 - alpha}.")
        if coverage > 1 - alpha + 0.10:
            print(
                f"Warning: Coverage {coverage} is much greater than"
                f" {1 - alpha}."
            )
