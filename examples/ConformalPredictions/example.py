import jax.numpy as np
import jax.random as jr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import parametricmatrixmodels as pmm

SEED = 0
rng = jr.key(SEED)

model = pmm.Model()

# simple NN with 2 outputs
hidden_dim = 16
hidden_layers = 2

for _ in range(hidden_layers):
    model += pmm.modules.LinearNN(k=hidden_dim)
    model += pmm.modules.Softplus()
model += pmm.modules.LinearNN(k=2)

# data is two simple functions
N_samples = 1000
x = np.linspace(-3, 3, N_samples).reshape(-1, 1)
y = np.hstack((x**2, x**3))
# add some noise
noisekey, rng = jr.split(rng)
y += 0.05 * jr.normal(noisekey, y.shape)

# split into train, val, cal, test
splits = [0.6, 0.2, 0.1, 0.1]
N_train = int(splits[0] * N_samples)
N_val = int(splits[1] * N_samples)
N_cal = int(splits[2] * N_samples)
N_test = N_samples - N_train - N_val - N_cal

# shuffle data
shufflekey, rng = jr.split(rng)

perm = jr.permutation(shufflekey, N_samples)

x_shuffled = x[perm]
y_shuffled = y[perm]

x_train, y_train = x_shuffled[:N_train], y_shuffled[:N_train]
x_val, y_val = (
    x_shuffled[N_train : N_train + N_val],
    y_shuffled[N_train : N_train + N_val],
)
x_cal, y_cal = (
    x_shuffled[N_train + N_val : N_train + N_val + N_cal],
    y_shuffled[N_train + N_val : N_train + N_val + N_cal],
)
x_test, y_test = (
    x_shuffled[N_train + N_val + N_cal :],
    y_shuffled[N_train + N_val + N_cal :],
)

x = np.linspace(-100, 100, 10 * N_samples).reshape(-1, 1)
y = np.hstack((x**2, x**3))

# standardize data
xscaler = StandardScaler()
yscaler = StandardScaler()
x_train = xscaler.fit_transform(x_train)
x_val = xscaler.transform(x_val)
x_cal = xscaler.transform(x_cal)
x_test = xscaler.transform(x_test)
x = xscaler.transform(x)
y_train = yscaler.fit_transform(y_train)
y_val = yscaler.transform(y_val)
y_cal = yscaler.transform(y_cal)
y_test = yscaler.transform(y_test)
y = yscaler.transform(y)

# compile the model
model.compile(rng, (1,))
# show the model
print(model)
# train the model
initkey, batchkey, rng = jr.split(rng, 3)
model.train(
    X=x_train,
    Y=y_train,
    X_val=x_val,
    Y_val=y_val,
    loss_fn="mse",
    lr=1e-3,
    batch_size=128,
    num_epochs=10000,
    convergence_threshold=0.0,
    early_stopping_patience=1000,
    early_stopping_tolerance=0.0,
    initialization_seed=initkey,
    batch_seed=batchkey,
)

# make predictions
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)
y_cal_pred = model.predict(x_cal)
y_test_pred = model.predict(x_test)
y_pred = model.predict(x)

# get gradients
# tuple of df/dp for each param, each of
# shape (N_samples, <output shape>, <param shape>)


def u(x_):
    dy_dparams = model.grad_params(x_)
    params = model.get_params()
    # array of df/dx of shape (N_samples, <output shape>, <input shape>)
    dy_dx = model.grad_input(x_, batched=True, max_batch_size=128)

    # l2 norm of the gradients, scaled by the magnitude of the parameters
    dy_dparams = np.array(
        [
            np.linalg.norm(
                (np.abs(p) * dp).reshape((dp.shape[0], 2, -1)), ord=2, axis=2
            )
            for p, dp in zip(params, dy_dparams)
        ]
    )
    # shape (N_params, N_samples, <output shape>), now sum over the params
    dy_dparams = np.sum(
        dy_dparams, axis=0
    )  # shape (N_samples, <output shape>)

    # l2 norm of dy/dx, scaled by the variance of the training inputs
    dy_dx = (
        np.linalg.norm(np.std(x_train) * dy_dx, ord=2, axis=(2,)) ** 2
    )  # shape (N_samples, <output shape>)

    # uncertainty heuristic
    return dy_dx + dy_dparams  # shape (N_samples, <output shape>)


# nonconformity scores
scores = np.abs(y_cal - y_cal_pred) / u(x_cal)

alpha = 0.1  # 90% prediction intervals

# score quantiles
qhat = np.quantile(scores, np.ceil((1 - alpha) * (N_cal + 1)) / N_cal)

# prediction intervals
muhat, stdhat = model.predict(x), u(x)
lower = muhat - qhat * stdhat
upper = muhat + qhat * stdhat

# unstandardize the data
x_train = xscaler.inverse_transform(x_train)
x_val = xscaler.inverse_transform(x_val)
x_cal = xscaler.inverse_transform(x_cal)
x_test = xscaler.inverse_transform(x_test)
x = xscaler.inverse_transform(x)
y_train = yscaler.inverse_transform(y_train)
y_val = yscaler.inverse_transform(y_val)
y_cal = yscaler.inverse_transform(y_cal)
y_test = yscaler.inverse_transform(y_test)
y = yscaler.inverse_transform(y)
y_train_pred = yscaler.inverse_transform(y_train_pred)
y_val_pred = yscaler.inverse_transform(y_val_pred)
y_cal_pred = yscaler.inverse_transform(y_cal_pred)
y_test_pred = yscaler.inverse_transform(y_test_pred)
y_pred = yscaler.inverse_transform(y_pred)

lower = yscaler.inverse_transform(lower)
upper = yscaler.inverse_transform(upper)


# plot
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
for i in range(y.shape[1]):
    axes[0, i].scatter(
        x_train, y_train[:, i], color="blue", label="train", s=5
    )
    axes[0, i].scatter(x_val, y_val[:, i], color="green", label="val", s=5)
    axes[0, i].scatter(x_cal, y_cal[:, i], color="orange", label="cal", s=5)
    axes[0, i].scatter(x_test, y_test[:, i], color="red", label="test", s=5)
    axes[0, i].scatter(x, y_pred[:, i], color="black", label="pred", s=1)

    axes[0, i].plot(
        x, lower[:, i], color="gray", linestyle="--", label="PI lower"
    )
    axes[0, i].plot(
        x, upper[:, i], color="gray", linestyle="--", label="PI upper"
    )

    axes[0, i].legend()

    # replot just the prediction and intervals,
    # but subtract off the expected value
    axes[1, i].plot(
        x,
        lower[:, i] - y_pred[:, i],
        color="gray",
        linestyle="--",
        label="PI lower",
    )
    axes[1, i].plot(
        x,
        upper[:, i] - y_pred[:, i],
        color="gray",
        linestyle="--",
        label="PI upper",
    )


plt.show()
