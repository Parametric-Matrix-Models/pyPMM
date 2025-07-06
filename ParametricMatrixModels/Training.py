import jax.numpy as np
from jax import grad, jit, lax
import jax
from functools import partial
import signal
import sys
from time import time
import random
from typing import Callable

"""
    Complex Adam Optimizer in fully compiled JAX.
"""


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


class ProgressBar:
    """
    Simple console progress bar
    """

    def __init__(self, total, length=40, extra_info=""):
        self.total = total
        self.length = length
        self.start(extra_info)

    def start(self, extra_info=""):
        self.last = 0
        self.starttime = time()  # estimate time remaining
        self.longest_str = 0
        self.extra_info = extra_info + (" | " if extra_info else "")

    def update(self, raw_progress, dynamic_info=""):
        if self.total <= 1e-9:
            return
        progress_frac = raw_progress / self.total
        progress_int = int(progress_frac * self.length)
        elapsed = time() - self.starttime
        est_total_time = elapsed / progress_frac if progress_frac > 1e-9 else 0
        remaining = est_total_time - elapsed
        progress = min(progress_int, self.length)
        if progress >= self.last:
            disp = (
                "\r"
                + self.extra_info
                + dynamic_info
                + (" " if dynamic_info else "")
                + "["
                + "#" * progress
                + " " * (self.length - progress)
                + "] ("
                + str(int(remaining))
                + "s)"
            )

            disp_len = len(disp)
            # pad with spaces to overwrite previous longest line
            diff = self.longest_str - disp_len
            if diff > 0:
                disp += " " * diff
            else:
                self.longest_str = disp_len

            sys.stdout.write(disp)
            sys.stdout.flush()
            self.last = progress

    def end(self, final_info=""):
        if self.total <= 1e-9:
            return
        disp = (
            f"\r{self.extra_info}"
            + final_info
            + (" " if final_info else "")
            + f"[{'#' * self.length}] ({int(time() - self.starttime)}s)"
        )

        disp_len = len(disp)
        # pad with spaces to overwrite previous longest line
        diff = self.longest_str - disp_len
        if diff > 0:
            disp += " " * diff
        else:
            self.longest_str = disp_len

        disp += "\n"
        sys.stdout.write(disp)
        sys.stdout.flush()


def make_schedule(
    scalar_or_schedule: float | Callable[[int], float]
) -> Callable[[int], float]:
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif np.ndim(scalar_or_schedule) == 0:
        return lambda _: scalar_or_schedule
    else:
        raise TypeError(type(scalar_or_schedule))


def adam(step_size, b1=0.9, b2=0.999, eps=1e-8, clip=np.inf):
    """
    Returns a function that computes the Adam update for real numbers.
    """

    step_size = make_schedule(step_size)

    def init(x0):
        """
        Initializes the Adam optimizer state.
        """
        m = np.zeros_like(x0, dtype=np.float32)
        v = np.zeros_like(x0, dtype=np.float32)
        return x0.astype(np.float32), m, v

    def update(i, dx, state):
        """
        Computes the Adam update for real numbers.
        """
        x, m, v = state

        # clip
        dx = np.clip(dx, -clip, clip)

        m = b1 * m + (1 - b1) * dx
        v = b2 * v + (1 - b2) * dx**2
        m_hat = m / (1 - b1 ** (i + 1))
        v_hat = v / (1 - b2 ** (i + 1))
        x = x - step_size(i) * m_hat / (np.sqrt(v_hat) + eps)
        return x, m, v

    def get_params(state):
        """
        Returns the parameters from the optimizer state.
        """
        params, _, _ = state
        return params

    def update_params_direct(new_params, state):
        """
        Updates the parameters directly in the optimizer state.
        """
        _, m, v = state
        return new_params, m, v

    return init, update, get_params, update_params_direct


def complex_adam(step_size, b1=0.9, b2=0.999, eps=1e-8, clip=np.inf):
    """
    Returns a function that computes the Adam update for complex numbers.
    """

    step_size = make_schedule(step_size)

    def init(x0):
        """
        Initializes the Adam optimizer state.
        """
        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        return x0, m, v

    def update(i, dx, state):
        """
        Computes the Adam update for complex numbers.
        """
        x, m, v = state

        # choose update based on dtype
        # this branch will be traced out since JAX arrays are static shape and
        # dtype
        if np.iscomplexobj(x):
            # conjugate and clip
            dx = np.clip(dx.real, -clip, clip) - 1j * np.clip(
                dx.imag, -clip, clip
            )

            m = b1 * m + (1 - b1) * dx
            v = b2 * v + (1 - b2) * (dx * np.conj(dx))
        else:
            # real numbers, clip
            dx = np.clip(dx, -clip, clip)

            m = b1 * m + (1 - b1) * dx
            v = b2 * v + (1 - b2) * dx**2

        m_hat = m / (1 - b1 ** (i + 1))
        v_hat = v / (1 - b2 ** (i + 1))
        x = x - (step_size(i) * m_hat / (np.sqrt(v_hat) + eps)).astype(x.dtype)

        return x, m, v

    def get_params(state):
        """
        Returns the parameters from the optimizer state.
        """
        params, _, _ = state
        return params

    def update_params_direct(new_params, state):
        """
        Updates the parameters directly in the optimizer state.
        """
        _, m, v = state
        return new_params, m, v

    return init, update, get_params, update_params_direct


def _train_step(
    update_fn,
    states,
    get_params,
    i,
    X_batch,
    Y_batch,
    Y_unc_batch,
    grad_loss_fn,
):
    """
    Performs a single training step.
    """

    # Compute gradients
    dparams = grad_loss_fn(
        X_batch, Y_batch, Y_unc_batch, tuple(map(get_params, states))
    )

    return tuple(map(partial(update_fn, i), dparams, states))


@partial(
    jit,
    static_argnames=(
        "update_fn",
        "get_params",
        "update_params_direct",
        "loss_fn",
        "grad_loss_fn",
        "batch_size",
        "start_epoch",
        "num_epochs",
        "convergence_threshold",
        "early_stopping_patience",
        "early_stopping_tolerance",
        "callback",
        "unroll",
        "verbose",
    ),
)
def _train(
    rng,
    update_fn,  # static, jittable
    states,
    get_params,  # static, jittable
    update_params_direct,  # static, jittable
    X,
    Y,
    Y_unc,  # uncertainty in the targets, if applicable
    X_val,
    Y_val,
    Y_val_unc,  # uncertainty in the validation targets, if applicable
    loss_fn,  # static, jittable
    grad_loss_fn,  # static, jittable
    batch_size,  # static [default should be the full dataset]
    start_epoch,  # static [default should be 0]
    num_epochs,  # static [default should be 100]
    convergence_threshold,  # static [default should be -np.inf (no convergence)]
    early_stopping_patience,  # static [default should be 10]
    early_stopping_tolerance,  # static [default should be -np.inf (no tolerance)],
    # advanced options
    callback,  # static, jittable (rngkey, step, params) -> rngkey, params
    unroll,  # static [default should be None, for unrolling the batch loop]
    verbose,  # static [default should be True]
):
    """
    Main training loop for the Adam optimizer. All non-jittable setup should be
    done before this function is called.

    Parameters
    ----------
    rng : jax.random.PRNGKey
        Random number generator key for JAX.
    update_fn : list of callables
        List of update functions for each parameter.
    states : list of tuples
        List of optimizer states for each parameter.
    get_params : callable
        Function that retrieves the parameters from the optimizer state.
    update_params_direct : callable
        Function that updates the parameters directly in the optimizer state.
    X : array-like
        (N_samples, (... feature dims ...)) training data.
    Y : array-like
        (N_samples, (... target dims ...)) training targets.
    X_val : array-like
        (N_samples, (... feature dims ...)) validation data.
    Y_val : array-like
        (N_samples, (... target dims ...)) validation targets.
    loss_fn : callable
        Function that computes the loss given the parameters and batch of data.
    grad_loss_fn : callable
        Function that computes the gradients of the loss with respect to the
        parameters.
    batch_size : int
        Size of the training batches.
    start_epoch : int
        Epoch to start training from. This is useful for resuming training.
    num_epochs : int
        Total number of epochs to train for.
    convergence_threshold : float
        Threshold for convergence. If the validation loss is ever below this
        value, training will stop early. Set to -np.inf to disable.
    early_stopping_patience : int
        Number of epochs to wait for improvement before stopping training
        early. Ensure early_stopping_patience >> validation_freq. Must be > 0.
        To disable early stopping set early_stopping_tolerance to -np.inf.
    early_stopping_tolerance : float
        Minimum improvement in validation loss required to reset the patience
        counter. If the validation loss does not improve by at least this much,
        the patience counter will be decremented. Set to -np.inf to disable
        early stopping.
    callback : callable
        Function that is called after each training step. It should take the
        current rngkey and epoch and the parameters as arguments and return the
        updated rngkey and parameters. This function must be jittable.
    unroll : int or None
        If not None, the number of batches to unroll in the batch loop. This
        can speed up training and increase compilation time. Default is no
        unrolling.
    verbose : bool
    """

    # since JAX arrays are static sizes, num_batches will be static
    num_batches = X.shape[0] // batch_size

    if verbose:
        killer = GracefulKiller()
        pb = ProgressBar(
            num_batches,
            length=20,
        )

    def start_progress_bar_callback(epoch):
        """
        Callback to start the progress bar at the beginning of each epoch.

        Will be entirely skipped if verbose is False.
        """
        pb.start(f"{epoch + 1}/{num_epochs}")
        return 0

    def update_progress_bar_callback(batch_idx):
        """
        Callback to update the progress bar after each batch.

        Will be entirely skipped if verbose is False.
        """
        pb.update(batch_idx)
        return 0

    def end_progress_bar_callback(val_loss, best_val_loss):
        """
        Callback to end the progress bar at the end of each epoch.

        Will be entirely skipped if verbose is False.
        """
        pb.end(f"{val_loss:.4e}/{best_val_loss:.4e}")
        return 0

    def batch_body_fn(batch_idx, batch_carry):
        """
        The part of the training loop that processes all batches in the dataset

        Each iteration of the body must execute in serial, and JAX will make
        sure of that, since batch_carry will update each loop, since it
        contains the Adam states.
        """

        shuffled_X, shuffled_Y, shuffled_Y_unc, states_, epoch = batch_carry

        if verbose:
            epoch += jax.pure_callback(
                update_progress_bar_callback, epoch, batch_idx
            )

        # shuffled_X and shuffled_Y are the pre-shuffled data
        #   for this epoch (constant)
        # states are the current optimizer states
        # epoch is the current epoch number (constant)

        X_batch = lax.dynamic_slice_in_dim(
            shuffled_X, batch_idx * batch_size, batch_size, axis=0
        )
        Y_batch = lax.dynamic_slice_in_dim(
            shuffled_Y, batch_idx * batch_size, batch_size, axis=0
        )
        Y_unc_batch = lax.dynamic_slice_in_dim(
            shuffled_Y_unc, batch_idx * batch_size, batch_size, axis=0
        )

        # Perform a single training step
        new_states = _train_step(
            update_fn,
            states_,
            get_params,
            epoch,
            X_batch,
            Y_batch,
            Y_unc_batch,
            grad_loss_fn,
        )

        return shuffled_X, shuffled_Y, shuffled_Y_unc, new_states, epoch

    def epoch_cond_callback(training_state):
        kill_now = killer.kill_now

        return not kill_now

    # training_state contains:
    # (rng, epoch, states, best_states, best_val_loss, patience)

    def epoch_cond_fn(training_state):
        """
        Continue while the epoch is less than num_epochs, the solution has not
        converged, and the patience has not run out.
        """

        if verbose:
            cont = jax.pure_callback(
                epoch_cond_callback, np.bool(True), training_state
            )
        else:
            cont = True

        rng, epoch, states, best_states, best_val_loss, patience = (
            training_state
        )
        return (
            cont
            & (epoch < num_epochs)
            & (best_val_loss > convergence_threshold)
            & (patience > 0)
        )

    def epoch_body_fn(training_state):
        """
        Iteration of the training loop for a single epoch. Handles shuffling,
        batching, validation, patience, and progress updates.
        """

        rng, epoch, states, best_states, best_val_loss, patience = (
            training_state
        )

        # new random key for this epoch
        rng, epoch_rng = jax.random.split(rng)

        # Shuffle the data for this epoch
        shuffled_X = jax.random.permutation(epoch_rng, X)
        shuffled_Y = jax.random.permutation(epoch_rng, Y)
        shuffled_Y_unc = jax.random.permutation(epoch_rng, Y_unc)

        # Initialize the progress bar
        if verbose:
            epoch += jax.pure_callback(
                start_progress_bar_callback, epoch, epoch
            )

        # Run the batch loop
        # since num_batches and X.shape[0] are static, one branch will be
        # completely traced out
        if X.shape[0] % batch_size == 0:
            # no remainder
            upper = num_batches

        batch_carry = (shuffled_X, shuffled_Y, shuffled_Y_unc, states, epoch)
        batch_carry = lax.fori_loop(
            0, num_batches, batch_body_fn, batch_carry, unroll=unroll
        )
        _, _, _, states, _ = batch_carry

        # deal with possible remainder, again this may be traced out
        if X.shape[0] % batch_size > 0:
            # handle the last batch
            X_batch = lax.dynamic_slice_in_dim(
                shuffled_X,
                num_batches * batch_size,
                X.shape[0] % batch_size,
                axis=0,
            )
            Y_batch = lax.dynamic_slice_in_dim(
                shuffled_Y,
                num_batches * batch_size,
                X.shape[0] % batch_size,
                axis=0,
            )
            Y_unc_batch = lax.dynamic_slice_in_dim(
                shuffled_Y_unc,
                num_batches * batch_size,
                X.shape[0] % batch_size,
                axis=0,
            )

            states = _train_step(
                update_fn,
                states,
                get_params,
                epoch,
                X_batch,
                Y_batch,
                Y_unc_batch,
                grad_loss_fn,
            )

        # Validation step
        val_loss = loss_fn(
            X_val, Y_val, Y_val_unc, tuple(map(get_params, states))
        )

        # patience handling
        # decrease patience if the validation loss has not improved
        improved = (
            val_loss <= best_val_loss - early_stopping_tolerance
        ).astype(np.int32)

        # linear update
        patience = (early_stopping_patience) * improved + (
            (patience - 1) * (1 - improved)
        )

        best_val_loss, best_states = lax.cond(
            val_loss < best_val_loss,
            lambda x, y: x,  # if the validation loss improved
            lambda x, y: y,  # if the validation loss did not improve
            (val_loss, states),
            (best_val_loss, best_states),
        )

        if verbose:
            epoch += jax.pure_callback(
                end_progress_bar_callback, epoch, val_loss, best_val_loss
            )

        # Call the callback function
        params = tuple(map(get_params, states))
        rng, params = callback(rng, epoch, params)
        states = tuple(map(update_params_direct, params, states))

        # Return the updated state for the next epoch
        return (
            rng,
            epoch + 1,  # increment epoch
            states,
            best_states,
            best_val_loss,
            patience,
        )

    # get initial validation loss
    val_loss = loss_fn(X_val, Y_val, Y_val_unc, tuple(map(get_params, states)))

    # Initial state for the training loop
    initial_state = (
        rng,
        start_epoch,
        states,
        states,  # best_states starts as the initial states
        val_loss,  # best_val_loss starts as the initial validation loss
        early_stopping_patience,  # patience starts at the configured value
    )

    # Run the training loop
    final_state = lax.while_loop(epoch_cond_fn, epoch_body_fn, initial_state)

    # Extract the best states and final epoch
    return (
        final_state[3],  # best states
        final_state[1],  # final epoch
    )


def train(
    init_params,
    loss_fn,
    X,
    Y=None,
    Y_unc=None,  # uncertainty in the targets, if applicable
    X_val=None,
    Y_val=None,
    Y_val_unc=None,  # uncertainty in the validation targets, if applicable
    lr=1e-3,
    batch_size=32,
    num_epochs=100,
    convergence_threshold=1e-12,
    early_stopping_patience=10,
    early_stopping_tolerance=1e-6,
    # advanced options
    callback=None,
    unroll=None,
    verbose=True,
    seed=None,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    clip=1e3,
    real=False,  # if True, use the real Adam optimizer
):
    """
    Train a model from scratch using the Adam optimizer.
    """
    # check if any of the data are double precision and give a warning if so
    # if any(
    #    (
    #        d is not None
    #        and (
    #            np.issubdtype(np.asarray(d).dtype, np.float64)
    #            or np.issubdtype(np.asarray(d).dtype, np.complex128)
    #        )
    #        for d in (X, Y, X_val, Y_val, Y_unc, Y_val_unc)
    #    )
    # ):
    #    print(
    #        "\033[1;91m[WARN] Some data are double precision. "
    #        "This may lead to significantly slower training on certain "
    #        "backends. It is strongly recommended to use single precision "
    #        "(float32/complex64) data for training.\033[0m"
    #    )

    # check if any parameters are double precision and give a warning if so
    if any(
        (
            np.issubdtype(np.asarray(param).dtype, np.float64)
            or np.issubdtype(np.asarray(param).dtype, np.complex128)
        )
        for param in init_params
    ):
        print(
            "\033[1;91m[WARN] Some parameters are double precision. "
            "This may lead to significantly slower training on certain "
            "backends. It is strongly recommended to use single precision "
            "(float32/complex64) parameters for training.\033[0m"
        )

    if callback is None:
        callback = lambda rng, step, params: (rng, params)

    if batch_size > X.shape[0]:
        print(
            f"Batch size {batch_size} is larger than the number of samples {X.shape[0]}. "
            "Using the full dataset as a single batch."
        )
        batch_size = X.shape[0]

    if Y is None and Y_val is not None:
        raise ValueError(
            "If Y_val is provided, Y must also be provided for supervised training."
        )

    if Y_unc is not None and Y is None:
        raise ValueError(
            "If Y_unc is provided, Y must also be provided for supervised training."
        )
    if Y_val_unc is not None and Y_val is None:
        raise ValueError(
            "If Y_val_unc is provided, Y_val must also be provided for supervised training."
        )
    if Y_unc is not None and Y_val is not None and Y_val_unc is None:
        raise ValueError(
            "If Y_unc and Y_val are provided, Y_val_unc must also be provided for supervised training."
        )

    # if no Y data is provided, assume unsupervised training
    # make Y data a dummy array with the same leading dimension as X
    # and redefine the loss function to take Y as a dummy variable
    if Y is None:
        Y = np.zeros((X.shape[0], 1), dtype=X.dtype)
        loss_fn_ = loss_fn
        loss_fn = lambda X, Y, params: loss_fn_(X, params)

    if Y_unc is None:
        # if no uncertainty in the targets is provided, assume it is 1
        Y_unc = np.ones_like(Y)
        loss_fn_ = loss_fn
        loss_fn = lambda X, Y, Y_unc, params: loss_fn_(X, Y, params)

        if Y_val is not None:
            Y_val_unc = np.ones_like(Y_val)

    # input handling
    # if validation data is not provided, use the training data
    if X_val is None:
        X_val = X
        Y_val = Y
        Y_val_unc = Y_unc

    # check sizes
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must have the same number of samples (first dimension)."
        )
    if Y_unc.shape != Y.shape:
        raise ValueError("Y_unc must have the same shape as Y.")
    if X_val.shape[0] != Y_val.shape[0]:
        raise ValueError(
            "X_val and Y_val must have the same number of samples (first dimension)."
        )
    if Y_val_unc.shape != Y_val.shape:
        raise ValueError("Y_val_unc must have the same shape as Y_val.")

    if X.shape[1:] != X_val.shape[1:]:
        raise ValueError(
            "X and X_val must have the same shape (except for the first dimension)."
        )
    if Y.shape[1:] != Y_val.shape[1:]:
        raise ValueError(
            "Y and Y_val must have the same shape (except for the first dimension)."
        )

    # set up everything for the JAX trainer
    grad_loss_fn = grad(loss_fn, argnums=3)

    # random key for JAX
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    rng = jax.random.key(seed)

    # Initialize the optimizer
    if real:
        init_fn, update_fn, get_params, update_params_direct = adam(
            step_size=lr,
            b1=b1,
            b2=b2,
            eps=eps,
            clip=clip,
        )
    else:
        # use complex Adam
        init_fn, update_fn, get_params, update_params_direct = complex_adam(
            step_size=lr,
            b1=b1,
            b2=b2,
            eps=eps,
            clip=clip,
        )

    states = tuple(map(init_fn, init_params))

    # train
    final_states, final_epoch = _train(
        rng,
        update_fn,
        states,
        get_params,
        update_params_direct,
        X,
        Y,
        Y_unc,
        X_val,
        Y_val,
        Y_val_unc,
        loss_fn,
        grad_loss_fn,
        batch_size=batch_size,
        start_epoch=0,  # always start from 0
        num_epochs=num_epochs,
        convergence_threshold=convergence_threshold,
        early_stopping_patience=early_stopping_patience,
        early_stopping_tolerance=early_stopping_tolerance,
        callback=callback,
        unroll=unroll,
        verbose=verbose,
    )

    # return the final parameters
    return tuple(map(get_params, final_states)), final_epoch, final_states


def make_loss_fn(fn_name: str, model_fn: Callable):
    """
    Create a loss function based on the model function.
    """
    if fn_name == "mse":

        def loss_fn(X, Y, params):
            Y_pred = model_fn(X, params)
            return np.mean(np.abs(Y_pred - Y) ** 2)

    elif fn_name == "mae":

        def loss_fn(X, Y, params):
            Y_pred = model_fn(X, params)
            return np.mean(np.abs(Y_pred - Y))

    elif fn_name == "mse_unc":
        # MSE with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params):
            Y_pred = model_fn(X, params)
            # Y_unc is assumed to be the uncertainty in the targets
            return np.mean(np.abs((Y_pred - Y) / Y_unc) ** 2)

    elif fn_name == "mae_unc":
        # MAE with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params):
            Y_pred = model_fn(X, params)
            # Y_unc is assumed to be the uncertainty in the targets
            return np.mean(np.abs((Y_pred - Y) / Y_unc))

    elif fn_name == "mre":
        # Mean relative error
        def loss_fn(X, Y, params):
            Y_pred = model_fn(X, params)
            return np.mean(np.abs((Y_pred - Y) / (Y + 1e-4)))

    elif fn_name == "mre_unc":
        # Mean relative error with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params):
            Y_pred = model_fn(X, params)
            return np.mean(np.abs((Y_pred - Y) / (Y + 1e-4) / Y_unc))

    elif fn_name == "mrd":
        # mean relative difference
        def loss_fn(X, Y, params):
            Y_pred = model_fn(X, params)
            return np.mean(np.abs((Y_pred - Y) / (2.0 * (Y + Y_pred) + 1e-4)))

    elif fn_name == "mrd_unc":
        # mean relative difference with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params):
            Y_pred = model_fn(X, params)
            return np.mean(
                np.abs((Y_pred - Y) / ((2.0 * (Y + Y_pred) + 1e-4) * Y_unc))
            )

    else:
        raise ValueError(f"Unknown loss function: {fn_name}")

    return loss_fn


if __name__ == "__main__":

    # test of a linear model
    # y = A @ x + b
    def model_fn_single(x, params):
        """
        Simple linear model: y = A @ X + b
        """
        A, b = params
        return lax.dot(A, x) + b

    model_fn = jax.vmap(model_fn_single, in_axes=(0, None))

    def loss_fn(X, Y, params):
        """
        Mean squared error loss function for the linear model.
        """
        Y_pred = model_fn(X, params)
        return np.mean(np.abs(Y_pred - Y) ** 2)

    # generate some random data
    N_samples = 10000
    N_features = 10
    N_targets = 5
    X = jax.random.normal(jax.random.key(0), (N_samples, N_features))
    target_A = jax.random.normal(jax.random.key(1), (N_targets, N_features))
    target_b = jax.random.normal(jax.random.key(2), (N_targets,))
    Y = (target_A @ X.T + target_b[:, None]).T  # (N_samples, N_targets)

    X_val = jax.random.normal(jax.random.key(3), (N_samples // 10, N_features))
    Y_val = (
        target_A @ X_val.T + target_b[:, None]
    ).T  # (N_samples // 10, N_targets)

    # initial parameters
    A_init = jax.random.normal(jax.random.key(4), (N_targets, N_features))
    b_init = jax.random.normal(jax.random.key(5), (N_targets,))

    params_init = (A_init, b_init)
    # train the model
    final_params, final_epoch, final_states = train(
        params_init,
        loss_fn,
        X,
        Y,
        X_val,
        Y_val,
        lr=1e-3,
        batch_size=32,
        num_epochs=1000,
        convergence_threshold=1e-12,
        early_stopping_patience=10,
        early_stopping_tolerance=1e-6,
        verbose=True,
    )

    # print the final error
    final_A, final_b = final_params
    final_Y_pred = model_fn(X_val, final_params)
    final_error = np.mean(
        (final_Y_pred.real - Y_val.real) ** 2
        + (final_Y_pred.imag - Y_val.imag) ** 2
    )
    print(f"Final error: {final_error:.4e}")
