import jax
import jax.numpy as np
import matplotlib.pyplot as plt

import parametricmatrixmodels as pmm

jax.config.update("jax_enable_x64", True)


def test_conformal():
    r"""
    Test ConformalizedModel
    """

    key = jax.random.key(0)

    nkey, pkey, key = jax.random.split(key, 3)

    # generate lightly noised abs(x) data
    N = 200
    x = np.linspace(-5, 5, N)
    y = np.abs(x) + 0.1 * jax.random.normal(nkey, (N,))

    # split into train and calibration sets
    n_train = 150
    x_shuffle = jax.random.permutation(pkey, x)
    y_shuffle = jax.random.permutation(pkey, y)
    x_train, y_train = x_shuffle[:n_train], y_shuffle[:n_train]
    x_calib, y_calib = x_shuffle[n_train:], y_shuffle[n_train:]

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
            activation=pmm.modules.ReLU(),
            init_magnitude=im,
        ),
        pmm.modules.LinearNN(
            out_features=8,
            bias=True,
            activation=pmm.modules.ReLU(),
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
        verbose=True,
    )

    # standard deviation of the training data
    std_x_train = np.std(x_train, axis=0)

    # make conformal model
    cmodel = pmm.conformal.ConformalizedModel(
        model, additional_data={"std_X_train": std_x_train}
    )

    # calibrate
    cmodel.calibrate(x_calib, y_calib)

    # make test data
    N_test = 500
    x_test = np.linspace(-6, 6, N_test)[:, None]
    y_test = np.abs(x_test)
    # get prediction intervals
    y_pred, (y_lower, y_upper) = cmodel(x_test)

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_test, y_test, label="True function", color="black")
    ax.scatter(
        x_calib,
        y_calib,
        label="Calibration data",
        color="red",
        alpha=0.5,
        s=20,
    )
    ax.plot(x_test, y_pred, label="NN Prediction", color="blue")
    ax.fill_between(
        x_test.flatten(),
        y_lower.flatten(),
        y_upper.flatten(),
        color="blue",
        alpha=0.2,
        label="Conformal Prediction Interval",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Conformalized Neural Network Regression")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    test_conformal()
