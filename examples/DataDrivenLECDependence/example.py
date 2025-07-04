import ParametricMatrixModels as PMM
import numpy as np
import matplotlib.pyplot as plt
import jax.random as jr

"""
    Example of building a two-module PMM in order to identify the LEC
    dependence of the ground state of a model from data alone.

    The idea is that if we don't know how the Hamiltonian should depend on the
    LECs, we can use a general ML model, in this case a regression PMM
    (AffineObservablePMM), to generate some features from the LECs, and a
    standard AffineEigenvaluePMM can use these features in an affine way to
    predict the ground state energy.

    The model looks like this:

    LECs -> AffineObservablePMM -> AffineEigenvaluePMM -> Ground State Energy

    or as a single equation for the final learned PMM Hamiltonian:

    M(cs) = M_0 + f_1(cs) * M_1 + f_2(c_s) * M_2 + ... + f_k(cs) * M_k

    where the f_i are the features learned by the AffineObservablePMM.
"""

# seed the random number generator for reproducibility
SEED = 0
np.random.seed(SEED)

# we start by defining the true underlying model, which will depend on some
# input parameters in a non-affine way


def features(c):
    return np.array([c[0] ** 2 - 0.5, np.sqrt(c[1] * c[2])])


def H(c, A, B, C):
    """
    H(c) = H0 + (c[0]**2 - 0.5) * H1 + sqrt(c[1] * c[2]) * H2
    """
    f = features(c)
    return A + f[0] * B + f[1] * C


# we define the true underlying operators, which are just some random matrices
n_true = 15
A = np.diag(np.random.randn(n_true))
B = np.random.randn(n_true, n_true) + 1j * np.random.randn(n_true, n_true)
B = (B + B.conj().T) / 2  # make it Hermitian
C = np.random.randn(n_true, n_true) + 1j * np.random.randn(n_true, n_true)
C = (C + C.conj().T) / 2  # make it Hermitian

# now we take some data from the true model, which we will use to train,
# validate, and test the PMM
N_samples = 1000
train_val_test_split = (0.8, 0.1, 0.1)
cs = np.random.uniform((-20, 0, 0), (20, 20, 20), size=(N_samples, 3))

# ground state energies
Es = np.array([np.linalg.eigvalsh(H(c, A, B, C))[0] for c in cs])
# add a dimension to the energies so that they are 2D arrays
Es = Es[:, np.newaxis]

# split the data into training, validation, and test sets
N_train = int(train_val_test_split[0] * N_samples)
N_val = int(train_val_test_split[1] * N_samples)
N_test = N_samples - N_train - N_val
cs_train = cs[:N_train]
cs_val = cs[N_train : N_train + N_val]
cs_test = cs[N_train + N_val :]
Es_train = Es[:N_train]
Es_val = Es[N_train : N_train + N_val]
Es_test = Es[N_train + N_val :]

# plot the data in 3D where the colors are the energies
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(
    cs_train[:, 0],
    cs_train[:, 1],
    cs_train[:, 2],
    c=Es_train,
    cmap="viridis",
    label="Train",
)
ax.scatter(
    cs_val[:, 0],
    cs_val[:, 1],
    cs_val[:, 2],
    c=Es_val,
    cmap="viridis",
    label="Validation",
    marker="x",
)
ax.scatter(
    cs_test[:, 0],
    cs_test[:, 1],
    cs_test[:, 2],
    c=Es_test,
    cmap="viridis",
    label="Test",
    marker="^",
)
ax.set_xlabel("c[0]")
ax.set_ylabel("c[1]")
ax.set_zlabel("c[2]")
ax.set_title("Data in 3D")
plt.colorbar(ax.collections[0], ax=ax, label="Energy")
plt.legend()
plt.show()

# we rescale the data using a UniformScaler so that all cs and Es are in the
# range [-1, 1]
scaler = PMM.Scalers.UniformScaler(clow=-1, chigh=1, Elow=-1, Ehigh=1)
cs_train_sc, Es_train_sc = scaler.fit_transform(cs_train, Es_train)
cs_val_sc, Es_val_sc = scaler.transform(cs_val, Es_val)
cs_test_sc, Es_test_sc = scaler.transform(cs_test, Es_test)

# now we build the PMM model
# since we don't know how the Hamiltonian should depend on the LECs,
# we don't know how many features we need, so this is a hyperparameter, k.
# of course, we know that the true model has 2 features
k = 2

# the size of the regression PMM and the size of the eigenvalue PMM need not be
# the same, so we can set them independently
n_regression = 5
n_eigenvalue = 5

# finally, we have the hyperparameters for the regression PMM, which determine
# the number of eigenvectors to use, r, and the number of observables to use,
# l.
r = 3
l = 2

# we now build the full model

model = PMM.Model()
model.append_module(
    PMM.Modules.AffineObservablePMM(
        n=n_regression,
        r=r,
        l=l,
        k=k,
        init_magnitude=1e-1,
    )
)
model.append_module(
    PMM.Modules.AffineEigenvaluePMM(
        n=n_eigenvalue,
        init_magnitude=1e-2,
    )
)

# we can print out the model both before and after preparing it
print("Model before preparation:")
print("-" * 50)
print(model)
print("-" * 50)

# we compile the model by giving it a JAX PRNG key and the shape of the input
# data.
# depending on your platform, you may see a warning about 64-bit precision, you
# can ignore this
model.compile(jr.key(SEED), cs_train.shape[1:])

print("Model after preparation:")
print("-" * 50)
print(model)
print("-" * 50)

# we can now train the model using the training data
model.train(
    cs_train_sc,
    Es_train_sc,
    cs_val_sc,
    Es_val_sc,
    loss_fn="mse",
    lr=1e-3,
    num_epochs=1000,
    seed=SEED,
    early_stopping_patience=100,
)

# now we can evaluate the model on the test data
test_Es_pred_sc = model(cs_test_sc)
# we can also evaluate the model on the training and validation data
train_Es_pred_sc = model(cs_train_sc)
val_Es_pred_sc = model(cs_val_sc)

# unscale the predictions
_, test_Es_pred = scaler.inverse_transform(cs_test_sc, test_Es_pred_sc)
_, train_Es_pred = scaler.inverse_transform(cs_train_sc, train_Es_pred_sc)
_, val_Es_pred = scaler.inverse_transform(cs_val_sc, val_Es_pred_sc)

# we can now plot the predictions against the true values
fig, ax = plt.subplots()

ax.scatter(Es_train, train_Es_pred, label="Train", alpha=0.5)
ax.scatter(Es_val, val_Es_pred, label="Validation", alpha=0.5, marker="x")
ax.scatter(Es_test, test_Es_pred, label="Test", alpha=0.5, marker="^")
Emin = min(np.min(Es_train), np.min(Es_val), np.min(Es_test))
Emax = max(np.max(Es_train), np.max(Es_val), np.max(Es_test))
ax.plot(
    [Emin, Emax],
    [Emin, Emax],
    color="black",
    linestyle="--",
    zorder=0,
)
ax.set_xlabel("True Energy")
ax.set_ylabel("Predicted Energy")
plt.show()

# we can even pop the Hamiltonian from the model and show what the learned
# features look like
model.pop_module()
model.compile(jr.key(SEED), cs_train.shape[1:])

# we can now get the learned features
f_train_pred_sc = model(cs_train_sc)
f_val_pred_sc = model(cs_val_sc)
f_test_pred_sc = model(cs_test_sc)

# and compare them to the true features
f_train = np.array([features(c) for c in cs_train])
f_val = np.array([features(c) for c in cs_val])
f_test = np.array([features(c) for c in cs_test])

# unscale by fitting a scaler to the true features
# f_scaler = PMM.Scalers.UniformScaler(clow=-1, chigh=1, Elow=-1, Ehigh=1)
# f_scaler.fit(cs_train, f_train)
#
# _, f_train_pred = f_scaler.inverse_transform(cs_train_sc, f_train_pred_sc)
# _, f_val_pred = f_scaler.inverse_transform(cs_val_sc, f_val_pred_sc)
# _, f_test_pred = f_scaler.inverse_transform(cs_test_sc, f_test_pred_sc)
f_train_pred = f_train_pred_sc
f_val_pred = f_val_pred_sc
f_test_pred = f_test_pred_sc


fig, axes = plt.subplots(1, f_train.shape[1] + 2, figsize=(12, 6))

for i in range(f_train.shape[1]):
    for j in range(k):
        axes[i].scatter(
            f_train[:, i],
            f_train_pred[:, j],
            label=f"Train Feature {j + 1}",
            alpha=0.5,
            color=f"C{j}",
        )
        axes[i].scatter(
            f_val[:, i],
            f_val_pred[:, j],
            label=f"Validation Feature {j + 1}",
            alpha=0.5,
            marker="x",
            color=f"C{j}",
        )
        axes[i].scatter(
            f_test[:, i],
            f_test_pred[:, j],
            label=f"Test Feature {j + 1}",
            alpha=0.5,
            marker="^",
            color=f"C{j}",
        )

cmaps = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"]

for j in range(k):
    axes[~1].scatter(
        f_train[:, 0],
        f_train[:, 1],
        c=f_train_pred[:, j],
        label=f"Train Feature {j + 1}",
        alpha=0.5,
        cmap=cmaps[j],
    )
    axes[~1].scatter(
        f_val[:, 0],
        f_val[:, 1],
        c=f_val_pred[:, j],
        label=f"Validation Feature {j + 1}",
        alpha=0.5,
        marker="x",
        cmap=cmaps[j],
    )
    axes[~1].scatter(
        f_test[:, 0],
        f_test[:, 1],
        c=f_test_pred[:, j],
        label=f"Test Feature {j + 1}",
        alpha=0.5,
        marker="^",
        cmap=cmaps[j],
    )
    axes[~0].scatter(
        cs_train[:, 0],
        cs_train[:, 1],
        c=f_train_pred[:, j],
        label=f"Train Feature {j + 1}",
        alpha=0.5,
        cmap=cmaps[j],
    )
    axes[~0].scatter(
        cs_val[:, 0],
        cs_val[:, 1],
        c=f_val_pred[:, j],
        label=f"Validation Feature {j + 1}",
        alpha=0.5,
        marker="x",
        cmap=cmaps[j],
    )
    axes[~0].scatter(
        cs_test[:, 0],
        cs_test[:, 1],
        c=f_test_pred[:, j],
        label=f"Test Feature {j + 1}",
        alpha=0.5,
        marker="^",
        cmap=cmaps[j],
    )

plt.show()
