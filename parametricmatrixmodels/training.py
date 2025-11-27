import random
import signal
import sys
import warnings
from dataclasses import dataclass
from functools import partial
from time import time

import jax
import jax.numpy as np
from beartype import beartype
from jax import grad, jit, lax
from jaxtyping import Array, Float, Inexact, Integer, jaxtyped

from . import tree_util as pmm_tree_util
from .model_util import ModelParams, ModelState
from .tree_util import (
    batch_leaves,
    random_permute_leaves,
    shapes_equal,
)
from .typing import (
    Any,
    Callable,
    Data,
    PyTree,
    Tuple,
    TypeAlias,
)

"""
    Complex Adam Optimizer in fully compiled JAX.
"""

# type alias for a single parameter array
ParamArray: TypeAlias = Inexact[Array, "..."]


# data class for OptimizerState (so that it becomes a leaf in PyTrees)
@jax.tree_util.register_dataclass
@dataclass
class OptimizerState:
    params: ParamArray
    m: ParamArray
    v: ParamArray

    def __iter__(self):
        return iter((self.params, self.m, self.v))


# training_state contains:
# (
#   batch_rng,
#   epoch,
#   adam_state,
#   best_adam_state,
#   model_state,
#   best_model_state,
#   model_rng,
#   best_model_rng,
#   best_val_loss,
#   best_epoch,
#   patience
# )
TrainingState: TypeAlias = Tuple[
    Any,
    int,
    OptimizerState,
    OptimizerState,
    ModelState,
    ModelState,
    Any,
    Any,
    float,
    int,
    int,
]


class GracefulKiller:
    kill_now = False

    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args) -> None:
        # if a second signal is received, exit immediately
        if self.kill_now:
            raise KeyboardInterrupt(
                "Multiple termination signals received, exiting."
            )
        self.kill_now = True


class ProgressBar:
    """
    Simple console progress bar
    """

    def __init__(
        self, total: int | float, length: int = 40, extra_info: str = ""
    ) -> None:
        self.total = total
        self.length = length
        self.start(extra_info)

    def start(self, extra_info: str = "") -> None:
        self.last = 0
        self.starttime = time()  # estimate time remaining
        self.longest_str = 0
        self.extra_info = extra_info + (" | " if extra_info else "")

    def update(
        self, raw_progress: int | float, dynamic_info: str = ""
    ) -> None:
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

    def end(self, final_info: str = "") -> None:
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


@jaxtyped(typechecker=beartype)
def make_schedule(
    scalar_or_schedule: float | Callable[[int], float],
) -> Callable[[int], float]:
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif np.ndim(scalar_or_schedule) == 0:
        return lambda _: scalar_or_schedule
    else:
        raise TypeError(type(scalar_or_schedule))


@jaxtyped(typechecker=beartype)
def adam(
    step_size: Callable[[int], float] | float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    clip: float = np.inf,
) -> Tuple[
    Callable[[ParamArray], OptimizerState],
    Callable[[int, float, OptimizerState], OptimizerState],
    Callable[[OptimizerState], ParamArray],
    Callable[[ParamArray, OptimizerState], OptimizerState],
]:
    """
    Returns functions that computes the Adam update for real numbers.
    """

    step_size = make_schedule(step_size)

    @jaxtyped(typechecker=beartype)
    def init(
        x0: ParamArray,
    ) -> OptimizerState:
        """
        Initializes the Adam optimizer state.
        """
        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        return OptimizerState(x0, m, v)

    @jaxtyped(typechecker=beartype)
    def update(
        i: int | Integer[Array, ""],
        dx: ParamArray,
        state: OptimizerState,
    ) -> OptimizerState:
        """
        Computes the Adam update for real numbers.
        """
        x, m, v = state

        # clip
        dx = np.clip(dx, -clip, clip)

        m = b1 * m + (1 - b1) * dx
        v = b2 * v + (1 - b2) * dx**2
        # explicit float32 to prevent upcasting in mixed precision
        m_hat = m / np.float32(1 - b1 ** (i + 1))
        v_hat = v / np.float32(1 - b2 ** (i + 1))
        x = x - step_size(i) * m_hat / (np.sqrt(v_hat) + eps)
        return OptimizerState(x, m, v)

    @jaxtyped(typechecker=beartype)
    def get_params(state: OptimizerState) -> ParamArray:
        """
        Returns the parameters from the optimizer state.
        """
        params, _, _ = state
        return params

    @jaxtyped(typechecker=beartype)
    def update_params_direct(
        new_params: ParamArray, state: OptimizerState
    ) -> OptimizerState:
        """
        Updates the parameters directly in the optimizer state.
        """
        _, m, v = state
        return OptimizerState(new_params, m, v)

    return init, update, get_params, update_params_direct


@jaxtyped(typechecker=beartype)
def complex_adam(
    step_size: Callable[[int], float] | float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    clip: float = np.inf,
) -> Tuple[
    Callable[[ParamArray], OptimizerState],
    Callable[[int, ParamArray, OptimizerState], OptimizerState],
    Callable[[OptimizerState], ParamArray],
    Callable[[ParamArray, OptimizerState], OptimizerState],
]:
    """
    Returns functions that computes the Adam update for complex numbers.
    """

    step_size = make_schedule(step_size)

    @jaxtyped(typechecker=beartype)
    def init(x0: ParamArray) -> OptimizerState:
        """
        Initializes the Adam optimizer state.
        """
        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        return OptimizerState(x0, m, v)

    @jaxtyped(typechecker=beartype)
    def update(
        i: int | Integer[Array, ""], dx: ParamArray, state: OptimizerState
    ) -> OptimizerState:
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

        # explicit float32 to prevent upcasting in mixed precision
        m_hat = m / np.float32(1 - b1 ** (i + 1))
        v_hat = v / np.float32(1 - b2 ** (i + 1))
        x = x - (step_size(i) * m_hat / (np.sqrt(v_hat) + eps))

        return OptimizerState(x, m, v)

    @jaxtyped(typechecker=beartype)
    def get_params(state: OptimizerState) -> ParamArray:
        """
        Returns the parameters from the optimizer state.
        """
        params, _, _ = state
        return params

    @jaxtyped(typechecker=beartype)
    def update_params_direct(
        new_params: ParamArray, state: OptimizerState
    ) -> OptimizerState:
        """
        Updates the parameters directly in the optimizer state.
        """
        _, m, v = state
        return OptimizerState(new_params, m, v)

    return init, update, get_params, update_params_direct


@jaxtyped(typechecker=beartype)
def _train_step(
    update_fn: Callable[[int, ParamArray, OptimizerState], OptimizerState],
    adam_states: PyTree[OptimizerState],
    get_params: Callable[[OptimizerState], ParamArray],
    model_states: ModelState,
    model_rng: Any,
    i: int | Integer[Array, ""],
    X_batch: Data,
    Y_batch: Data | None,
    Y_unc_batch: Data | None,
    grad_loss_fn: Callable[
        [
            Data,
            Data | None,
            Data | None,
            ModelParams,
            bool,
            ModelState,
            Any,
        ],
        Tuple[ModelParams, ModelState],
    ],
):
    """
    Performs a single training step.
    """

    # split the model rng for this step
    new_model_rng, model_rng = jax.random.split(model_rng)

    # Compute gradients
    dparams, new_states = grad_loss_fn(
        X_batch,
        Y_batch,
        Y_unc_batch,
        jax.tree.map(
            get_params,
            adam_states,
            is_leaf=lambda x: isinstance(x, OptimizerState),
        ),
        True,  # training mode
        model_states,
        model_rng,
    )

    return (
        jax.tree.map(
            lambda dp, a_s: update_fn(i, dp, a_s),
            dparams,
            adam_states,
            is_leaf=lambda x: isinstance(x, OptimizerState),
        ),
        new_states,
        new_model_rng,
    )


@partial(
    jit,
    static_argnames=(
        "update_fn",
        "get_params",
        "update_params_direct",
        "loss_fn",
        "grad_loss_fn",
        "batch_size",
        "val_batch_size",
        "start_epoch",
        "epochs",
        "target_loss",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "callback",
        "unroll",
        "verbose",
    ),
)
@jaxtyped(typechecker=beartype)
def _train(
    batch_rng: Any,
    update_fn: Callable[
        [int, ParamArray, OptimizerState], OptimizerState
    ],  # static, jittable
    adam_state: PyTree[OptimizerState],
    get_params: Callable[[OptimizerState], ParamArray],  # static, jittable
    update_params_direct: Callable[
        [ParamArray, OptimizerState], OptimizerState
    ],  # static, jittable
    init_state: ModelState,  # initial model state
    init_rng: Any,  # initial model rng
    X: Data,
    Y: Data | None,
    Y_unc: Data | None,  # uncertainty in the targets, if applicable
    X_val: Data,
    Y_val: Data | None,
    Y_val_unc: (
        Data | None
    ),  # uncertainty in the validation targets, if applicable
    loss_fn: Callable[
        [Data, Data, Data, ModelParams, bool, ModelState, Any],
        Tuple[float, ModelState],
    ],  # static, jittable
    grad_loss_fn: Callable[
        [Data, Data, Data, ModelParams, bool, ModelState, Any],
        Tuple[ModelParams, ModelState],
    ],  # static, jittable
    batch_size: int,  # static [default should be the full dataset]
    val_batch_size: int,  # static [default should be the full val dataset]
    start_epoch: int,  # static [default should be 0]
    epochs: int,  # static [default should be 100]
    target_loss: float,  # static [default should be -np.inf]
    early_stopping_patience: int,  # static [default should be 100]
    early_stopping_min_delta: float,  # static [default should be -np.inf]
    # advanced options
    callback: Callable[
        [Any, int, ModelParams], Tuple[Any, ModelParams]
    ],  # static, jittable
    unroll: (
        int | None
    ),  # static [default should be None, for unrolling the batch loop]
    verbose: bool,  # static [default should be True]
):
    """
    Main training loop for the Adam optimizer. All non-jittable setup should be
    done before this function is called.

    Parameters
    ----------
    loss_fn : callable
        Function that computes the loss given the parameters and batch of data.
    grad_loss_fn : callable
        Function that computes the gradients of the loss with respect to the
        parameters.
    batch_size : int
        Size of the training batches.
    val_batch_size : int
        Size of the validation batches.
    start_epoch : int
        Epoch to start training from. This is useful for resuming training.
    epochs : int
        Total number of epochs to train for.
    target_loss : float
        Threshold for convergence. If the validation loss is ever below this
        value, training will stop early. Set to -np.inf to disable.
    early_stopping_patience : int
        Number of epochs to wait for improvement before stopping training
        early. Ensure early_stopping_patience >> validation_freq. Must be > 0.
        To disable early stopping set early_stopping_tolerance to -np.inf.
    early_stopping_min_delta : float
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
    orig_batch_size = jax.tree.leaves(X)[0].shape[0]
    orig_val_batch_size = jax.tree.leaves(X_val)[0].shape[0]
    num_batches = orig_batch_size // batch_size
    num_val_batches = orig_val_batch_size // val_batch_size
    batch_remainder = orig_batch_size % batch_size
    val_batch_remainder = orig_val_batch_size % val_batch_size

    total_val_batches = num_val_batches + (1 if val_batch_remainder > 0 else 0)

    if verbose:
        killer = GracefulKiller()
        pb = ProgressBar(
            num_batches,
            length=20,
        )

    # batch the loss function for validation
    @jaxtyped(typechecker=beartype)
    def batched_val_loss_fn(
        X: Data,
        Y: Data | None,
        Y_unc: Data | None,
        params: ModelParams,
        state: ModelState,
        model_rng: Any,
        epoch_rng: Any,
    ) -> float | Float[Array, ""]:
        @jaxtyped(typechecker=beartype)
        def batch_val_body_fn(
            batch_idx: int | Integer[Array, ""],
            batch_carry: Tuple[
                Data, Data | None, Data | None, float | Float[Array, ""]
            ],
        ) -> Tuple[Data, Data | None, Data | None, float | Float[Array, ""]]:
            (
                shuffled_val_X,
                shuffled_val_Y,
                shuffled_val_Y_unc,
                mean_val_loss,
            ) = batch_carry

            # since all Data are arbitrary PyTrees, we slice each leaf
            X_val_batch = batch_leaves(
                shuffled_val_X, val_batch_size, batch_idx, axis=0
            )
            Y_val_batch = (
                batch_leaves(shuffled_val_Y, val_batch_size, batch_idx, axis=0)
                if shuffled_val_Y is not None
                else None
            )
            Y_unc_val_batch = (
                batch_leaves(
                    shuffled_val_Y_unc, val_batch_size, batch_idx, axis=0
                )
                if shuffled_val_Y_unc is not None
                else None
            )

            # Compute the loss for this batch
            # do not update state or rng in validation mode
            val_loss, _ = loss_fn(
                X_val_batch,
                Y_val_batch,
                Y_unc_val_batch,
                params,
                False,  # validation mode
                state,
                model_rng,
            )

            return (
                shuffled_val_X,
                shuffled_val_Y,
                shuffled_val_Y_unc,
                mean_val_loss + val_loss / total_val_batches,
            )

        # shuffle the validation data
        # since all Data are arbitrary PyTrees, we map the permutation over
        # the leaves
        # so long as the key isn't changed, the permutation will be the same
        # for all leaves
        shuffled_X_val = random_permute_leaves(X_val, epoch_rng, axis=0)
        shuffled_Y_val = (
            random_permute_leaves(Y_val, epoch_rng, axis=0)
            if Y_val is not None
            else None
        )
        shuffled_Y_unc_val = (
            random_permute_leaves(Y_val_unc, epoch_rng, axis=0)
            if Y_val_unc is not None
            else None
        )

        # scan over the validation batches
        _, _, _, mean_val_loss = lax.fori_loop(
            0,
            num_val_batches,
            batch_val_body_fn,
            (
                shuffled_X_val,
                shuffled_Y_val,
                shuffled_Y_unc_val,
                0.0,  # initial mean validation loss
            ),
            unroll=unroll,
        )

        # deal with possible remainder
        if val_batch_remainder > 0:
            # handle the last batch
            X_val_batch = batch_leaves(
                shuffled_X_val,
                val_batch_size,
                num_val_batches,
                length=val_batch_remainder,
                axis=0,
            )
            Y_val_batch = (
                batch_leaves(
                    shuffled_Y_val,
                    val_batch_size,
                    num_val_batches,
                    length=val_batch_remainder,
                    axis=0,
                )
                if shuffled_Y_val is not None
                else None
            )
            Y_unc_val_batch = (
                batch_leaves(
                    shuffled_Y_unc_val,
                    val_batch_size,
                    num_val_batches,
                    length=val_batch_remainder,
                    axis=0,
                )
                if shuffled_Y_unc_val is not None
                else None
            )

            # Compute the loss for this batch
            # do not update state or rng in validation mode
            val_loss, _ = loss_fn(
                X_val_batch,
                Y_val_batch,
                Y_unc_val_batch,
                params,
                False,  # validation mode
                state,
                model_rng,
            )

            # add the last batch loss to the mean
            mean_val_loss += val_loss / total_val_batches

        # return the mean validation loss
        return mean_val_loss

    def start_progress_bar_callback(epoch: int) -> int:
        """
        Callback to start the progress bar at the beginning of each epoch.

        Will be entirely skipped if verbose is False.
        """
        pb.start(f"{epoch + 1}/{epochs}")
        return 0

    def update_progress_bar_callback(batch_idx: int) -> int:
        """
        Callback to update the progress bar after each batch.

        Will be entirely skipped if verbose is False.
        """
        pb.update(batch_idx)
        return 0

    def end_progress_bar_callback(
        val_loss: float, best_val_loss: float
    ) -> int:
        """
        Callback to end the progress bar at the end of each epoch.

        Will be entirely skipped if verbose is False.
        """
        pb.end(f"{val_loss:.4e}/{best_val_loss:.4e}")
        return 0

    @jaxtyped(typechecker=beartype)
    def batch_body_fn(
        batch_idx: int | Integer[Array, ""],
        batch_carry: Tuple[
            Data,
            Data | None,
            Data | None,
            PyTree[OptimizerState, " O"],
            ModelState,
            Any,
            int | Integer[Array, ""],
        ],
    ) -> Tuple[
        Data,
        Data | None,
        Data | None,
        PyTree[OptimizerState, " O"],
        ModelState,
        Any,
        int | Integer[Array, ""],
    ]:
        """
        The part of the training loop that processes all batches in the dataset

        Each iteration of the body must execute in serial, and JAX will make
        sure of that, since batch_carry will update each loop, since it
        contains the Adam state.
        """

        (
            shuffled_X,
            shuffled_Y,
            shuffled_Y_unc,
            adam_state_,
            model_state_,
            model_rng_,
            epoch,
        ) = batch_carry

        if verbose:
            epoch += jax.pure_callback(
                update_progress_bar_callback, epoch, batch_idx
            )

        # shuffled_X and shuffled_Y are the pre-shuffled data
        #   for this epoch (constant)
        # adam_state are the current optimizer state
        # epoch is the current epoch number (constant)

        # since all Data are arbitrary PyTrees, we slice each leaf
        X_batch = batch_leaves(shuffled_X, batch_size, batch_idx, axis=0)
        Y_batch = (
            batch_leaves(shuffled_Y, batch_size, batch_idx, axis=0)
            if Y is not None
            else None
        )
        Y_unc_batch = (
            batch_leaves(shuffled_Y_unc, batch_size, batch_idx, axis=0)
            if Y_unc is not None
            else None
        )

        # Perform a single training step
        new_adam_state, new_model_state, new_model_rng = _train_step(
            update_fn,
            adam_state_,
            get_params,
            model_state_,
            model_rng_,
            epoch,
            X_batch,
            Y_batch,
            Y_unc_batch,
            grad_loss_fn,
        )

        return (
            shuffled_X,
            shuffled_Y,
            shuffled_Y_unc,
            new_adam_state,
            new_model_state,
            new_model_rng,
            epoch,
        )

    def epoch_cond_callback(training_state: TrainingState) -> bool:
        kill_now = killer.kill_now

        return not kill_now

    # training_state contains:
    # (
    #   batch_rng,
    #   epoch,
    #   adam_state,
    #   best_adam_state,
    #   model_state,
    #   best_model_state,
    #   model_rng,
    #   best_model_rng,
    #   best_val_loss,
    #   best_epoch,
    #   patience
    # )

    def epoch_cond_fn(training_state: TrainingState) -> bool:
        """
        Continue while the epoch is less than epochs, the solution has not
        converged, and the patience has not run out.
        """

        if verbose:
            cont = jax.pure_callback(
                epoch_cond_callback, np.bool(True), training_state
            )
        else:
            cont = True

        (
            batch_rng,
            epoch,
            adam_state,
            best_adam_state,
            model_state,
            best_model_state,
            model_rng,
            best_model_rng,
            best_val_loss,
            best_epoch,
            patience,
        ) = training_state
        return (
            cont
            & (epoch < epochs)
            & (best_val_loss > target_loss)
            & (patience > 0)
        )

    def epoch_body_fn(training_state: TrainingState) -> TrainingState:
        """
        Iteration of the training loop for a single epoch. Handles shuffling,
        batching, validation, patience, and progress updates.
        """

        (
            batch_rng,
            epoch,
            adam_state,
            best_adam_state,
            model_state,
            best_model_state,
            model_rng,
            best_model_rng,
            best_val_loss,
            best_epoch,
            patience,
        ) = training_state

        # new random key for this epoch
        batch_rng, epoch_rng = jax.random.split(batch_rng)

        # Shuffle the data for this epoch
        shuffled_X = random_permute_leaves(X, epoch_rng, axis=0)
        shuffled_Y = (
            random_permute_leaves(Y, epoch_rng, axis=0)
            if Y is not None
            else None
        )
        shuffled_Y_unc = (
            random_permute_leaves(Y_unc, epoch_rng, axis=0)
            if Y_unc is not None
            else None
        )

        # Initialize the progress bar
        if verbose:
            epoch += jax.pure_callback(
                start_progress_bar_callback, epoch, epoch
            )

        # Run the batch loop

        batch_carry = (
            shuffled_X,
            shuffled_Y,
            shuffled_Y_unc,
            adam_state,
            model_state,
            model_rng,
            epoch,
        )
        batch_carry = lax.fori_loop(
            0, num_batches, batch_body_fn, batch_carry, unroll=unroll
        )
        _, _, _, adam_state, model_state, model_rng, _ = batch_carry

        # deal with possible remainder, again this may be traced out
        if batch_remainder > 0:
            # handle the last batch
            X_batch = batch_leaves(
                shuffled_X,
                batch_size,
                num_batches,
                length=batch_remainder,
                axis=0,
            )
            Y_batch = (
                batch_leaves(
                    shuffled_Y,
                    batch_size,
                    num_batches,
                    length=batch_remainder,
                    axis=0,
                )
                if shuffled_Y is not None
                else None
            )
            Y_unc_batch = (
                batch_leaves(
                    shuffled_Y_unc,
                    batch_size,
                    num_batches,
                    length=batch_remainder,
                    axis=0,
                )
                if shuffled_Y_unc is not None
                else None
            )

            adam_state, model_state, model_rng = _train_step(
                update_fn,
                adam_state,
                get_params,
                model_state,
                model_rng,
                epoch,
                X_batch,
                Y_batch,
                Y_unc_batch,
                grad_loss_fn,
            )

        # Validation step
        batch_rng, epoch_rng = jax.random.split(batch_rng)
        val_loss = batched_val_loss_fn(
            X_val,
            Y_val,
            Y_val_unc,
            jax.tree.map(
                get_params,
                adam_state,
                is_leaf=lambda x: isinstance(x, OptimizerState),
            ),
            model_state,
            model_rng,
            epoch_rng,
        )

        # patience handling
        # decrease patience if the validation loss has not improved
        improved = (
            val_loss <= best_val_loss - early_stopping_min_delta
        ).astype(np.int32)

        # linear update
        patience = (early_stopping_patience) * improved + (
            (patience - 1) * (1 - improved)
        )

        (
            best_val_loss,
            best_epoch,
            best_adam_state,
            best_model_state,
            best_model_rng,
        ) = lax.cond(
            val_loss < best_val_loss,
            lambda x, y: x,  # if the validation loss improved
            lambda x, y: y,  # if the validation loss did not improve
            (val_loss, epoch, adam_state, model_state, model_rng),
            (
                best_val_loss,
                best_epoch,
                best_adam_state,
                best_model_state,
                best_model_rng,
            ),
        )

        if verbose:
            epoch += jax.pure_callback(
                end_progress_bar_callback, epoch, val_loss, best_val_loss
            )

        # Call the callback function
        params = jax.tree.map(
            get_params,
            adam_state,
            is_leaf=lambda x: isinstance(x, OptimizerState),
        )
        batch_rng, params = callback(batch_rng, epoch, params)
        adam_state = jax.tree.map(
            update_params_direct,
            params,
            adam_state,
            is_leaf=lambda x: isinstance(x, OptimizerState),
        )

        # Return the updated state for the next epoch
        return (
            batch_rng,
            epoch + 1,  # increment epoch
            adam_state,
            best_adam_state,
            model_state,
            best_model_state,
            model_rng,
            best_model_rng,
            best_val_loss,
            best_epoch,
            patience,
        )

    # get initial validation loss
    # val_loss, _ = loss_fn(
    #    X_val,
    #    Y_val,
    #    Y_val_unc,
    #    tuple(map(get_params, adam_state)),
    #    False,  # validation mode
    #    init_state,
    #    init_rng,
    # )
    init_rng, epoch_rng = jax.random.split(init_rng)
    val_loss = batched_val_loss_fn(
        X_val,
        Y_val,
        Y_val_unc,
        jax.tree.map(
            get_params,
            adam_state,
            is_leaf=lambda x: isinstance(x, OptimizerState),
        ),
        init_state,
        init_rng,
        epoch_rng,
    )

    # Initial state for the training loop
    initial_while_state = (
        batch_rng,
        start_epoch,
        adam_state,
        adam_state,  # best_state starts as the initial state
        init_state,  # initial model state
        init_state,  # best model state starts as the initial state
        init_rng,  # initial model rng
        init_rng,  # best model rng starts as the initial rng
        val_loss,  # best_val_loss starts as the initial validation loss
        start_epoch,  # best epoch
        early_stopping_patience,  # patience starts at the configured value
    )

    # Run the training loop
    final_while_state = lax.while_loop(
        epoch_cond_fn, epoch_body_fn, initial_while_state
    )

    if verbose:
        jax.debug.print(
            "\n"
            + 40 * "="
            + "\nTotal epochs: {epochs}"
            "\n(best epoch: {best_epoch})"
            "\n(best validation loss: {best_val_loss:.4E})"
            "\n"
            + 40 * "=",
            epochs=final_while_state[1],
            best_epoch=final_while_state[9],
            best_val_loss=final_while_state[8],
        )

    # Extract the best adam state, model state, model rng, and final epoch
    return (
        final_while_state[3],  # best adam_state
        final_while_state[5],  # best model state
        final_while_state[7],  # best model rng
        final_while_state[1],  # final epoch
        final_while_state[8],  # best validation loss
        final_while_state[9],  # best epoch
    )


@jaxtyped(typechecker=beartype)
def train(
    init_params: ModelParams,  # model parameters
    init_state: ModelState,  # model state
    init_rng: Any | int | None,  # model rng
    loss_fn: (
        # supervised with uncertainty
        Callable[
            [Data, Data, Data, ModelParams, bool, ModelState, Any],
            Tuple[float, ModelState],
        ]
        # unsupervised
        | Callable[
            [Data, ModelParams, bool, ModelState, Any],
            Tuple[float, ModelState],
        ]
        # supervised without uncertainty
        | Callable[
            [Data, Data, ModelParams, bool, ModelState, Any],
            Tuple[float, ModelState],
        ]
    ),  # loss function, three different signatures supported
    X: PyTree[
        Inexact[Array, "num_samples ?*features"], " InStruct"
    ],  # in features
    Y: (
        PyTree[Inexact[Array, "num_samples ?*targets"], " OutStruct"] | None
    ) = None,  # targets
    Y_unc: (
        PyTree[Inexact[Array, "num_samples ?*targets"], " OutStruct"] | None
    ) = None,  # uncertainty in the targets, if applicable
    X_val: (
        PyTree[Inexact[Array, "num_val_samples ?*features"], " InStruct"]
        | None
    ) = None,  # validation in features
    Y_val: (
        PyTree[Inexact[Array, "num_val_samples ?*targets"], " OutStruct"]
        | None
    ) = None,  # validation targets
    Y_val_unc: (
        PyTree[Inexact[Array, "num_val_samples ?*targets"], " OutStruct"]
        | None
    ) = None,  # uncertainty in the validation targets, if applicable
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 100,
    target_loss: float = 1e-12,
    early_stopping_patience: int = 100,
    early_stopping_min_delta: float = -np.inf,
    # advanced options
    callback: (
        Callable[[Any, int, ModelParams], Tuple[Any, ModelParams]] | None
    ) = None,
    unroll: int | None = None,
    verbose: bool = True,
    batch_rng: Any | int | None = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    clip: float = 1e3,
    real: bool = False,  # if True, use the real Adam optimizer
):
    """
    Train a model from scratch using the Adam optimizer.
    """
    # check if any parameters are double precision and give a warning if so
    if any(
        (
            np.issubdtype(np.asarray(param).dtype, np.float64)
            or np.issubdtype(np.asarray(param).dtype, np.complex128)
        )
        for param in jax.tree.leaves(init_params)
    ):
        warnings.warn(
            "Some parameters are double precision. "
            "This may lead to significantly slower training on certain "
            "backends. It is strongly recommended to use single precision "
            "(float32/complex64) parameters for training.",
            UserWarning,
        )

    # check if data is double precision and give a warning if so
    if (
        any(
            (
                np.issubdtype(np.asarray(data).dtype, np.float64)
                or np.issubdtype(np.asarray(data).dtype, np.complex128)
            )
            for data in jax.tree.leaves(X)
        )
        or (
            Y is not None
            and any(
                (
                    np.issubdtype(np.asarray(data).dtype, np.float64)
                    or np.issubdtype(np.asarray(data).dtype, np.complex128)
                )
                for data in jax.tree.leaves(Y)
            )
        )
        or (
            X_val is not None
            and any(
                (
                    np.issubdtype(np.asarray(data).dtype, np.float64)
                    or np.issubdtype(np.asarray(data).dtype, np.complex128)
                )
                for data in jax.tree.leaves(X_val)
            )
        )
        or (
            Y_val is not None
            and any(
                (
                    np.issubdtype(np.asarray(data).dtype, np.float64)
                    or np.issubdtype(np.asarray(data).dtype, np.complex128)
                )
                for data in jax.tree.leaves(Y_val)
            )
        )
        or (
            Y_val_unc is not None
            and any(
                (
                    np.issubdtype(np.asarray(data).dtype, np.float64)
                    or np.issubdtype(np.asarray(data).dtype, np.complex128)
                )
                for data in jax.tree.leaves(Y_val_unc)
            )
        )
    ):
        warnings.warn(
            "Some data is double precision. "
            "This may lead to significantly slower training on certain "
            "backends. It is strongly recommended to use single precision "
            "(float32/complex64) data for training.",
            UserWarning,
        )

    if callback is None:

        def callback(
            rng: Any, step: int, params: ModelParams
        ) -> Tuple[Any, ModelParams]:
            """
            Dummy callback that does nothing.
            """
            return rng, params

    num_samples = jax.tree.leaves(X)[0].shape[0]

    if batch_size > num_samples:
        print(
            f"Batch size {batch_size} is larger than the number of samples "
            f"{num_samples}. Using the full dataset as a single batch."
        )
        batch_size = num_samples

    if Y is None and Y_val is not None:
        raise ValueError(
            "If Y_val is provided, Y must also be provided for supervised "
            "training."
        )
    if Y_unc is not None and Y is None:
        raise ValueError(
            "If Y_unc is provided, Y must also be provided for supervised"
            " training."
        )
    if Y_val_unc is not None and Y_val is None:
        raise ValueError(
            "If Y_val_unc is provided, Y_val must also be provided for"
            " supervised training."
        )
    if Y_unc is not None and Y_val is not None and Y_val_unc is None:
        raise ValueError(
            "If Y_unc and Y_val are provided, Y_val_unc must also be provided"
            " for supervised training."
        )

    # if no Y data is provided, assume unsupervised training
    # make Y data a dummy array with the same leading dimension as X
    # and redefine the loss function to take Y as a dummy variable
    if Y is None:
        loss_fn_unsupervised = loss_fn

        @jaxtyped(typechecker=beartype)
        def loss_fn(
            X: Data,
            Y: None,
            params: ModelParams,
            training: bool,
            state: ModelState,
            rng: Any,
        ) -> Tuple[float | Float[Array, ""], ModelState]:
            """
            Wrapper for the loss function that ignores Y.
            """
            return loss_fn_unsupervised(X, params, training, state, rng)

    if Y_unc is None:
        loss_fn_no_unc = loss_fn

        @jaxtyped(typechecker=beartype)
        def loss_fn(
            X: Data,
            Y: Data,
            Y_unc: None,
            params: ModelParams,
            training: bool,
            state: ModelState,
            rng: Any,
        ) -> Tuple[float | Float[Array, ""], ModelState]:
            """
            Wrapper for the loss function that ignores Y_unc.
            """
            return loss_fn_no_unc(X, Y, params, training, state, rng)

        if Y_val is not None:
            Y_val_unc = None

    # input handling
    # if validation data is not provided, use the training data
    if X_val is None:
        X_val = X
        Y_val = Y
        Y_val_unc = Y_unc

    # check sizes
    # this should've already been caught by jaxtyping, but just in case
    if Y is not None and not shapes_equal(X, Y, axes=0):
        raise ValueError(
            "X and Y must have the same number of samples (first dimension)."
        )
    if Y is not None and Y_unc is not None and not shapes_equal(Y, Y_unc):
        raise ValueError("Y_unc must have the same shape as Y.")
    if (
        X_val is not None
        and Y_val is not None
        and not shapes_equal(X_val, Y_val, axes=0)
    ):
        raise ValueError(
            "X_val and Y_val must have the same number of samples (first"
            " dimension)."
        )
    if (
        Y_val is not None
        and Y_val_unc is not None
        and not shapes_equal(Y_val, Y_val_unc)
    ):
        raise ValueError("Y_val_unc must have the same shape as Y_val.")
    if not shapes_equal(X, X_val, axes=slice(1, None)):
        raise ValueError(
            "X and X_val must have the same shape (except for the first"
            " dimension)."
        )
    if Y is not None and not shapes_equal(Y, Y_val, axes=slice(1, None)):
        raise ValueError(
            "Y and Y_val must have the same shape (except for the first"
            " dimension)."
        )

    # set up everything for the JAX trainer
    # has_aux=True allows us to return the new state from the loss function
    # this will return (gradients, new_state)
    grad_loss_fn = grad(loss_fn, argnums=3, has_aux=True)

    # random key for JAX
    if batch_rng is None:
        batch_rng = jax.random.key(random.randint(0, 2**32 - 1))
    if isinstance(batch_rng, int):
        batch_rng = jax.random.key(batch_rng)

    # random key for the model itself
    if isinstance(init_rng, int):
        init_rng = jax.random.key(init_rng)
    elif init_rng is None:
        init_rng = jax.random.key(random.randint(0, 2**32 - 1))

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

    adam_state = jax.tree.map(init_fn, init_params)

    # make sure the validation batch size isn't larger than the validation set
    num_val_samples = jax.tree.leaves(X_val)[0].shape[0]
    if num_val_samples < batch_size:
        val_batch_size = num_val_samples
        warnings.warn(
            f"Validation batch size {batch_size} is larger than the number of"
            f" validation samples {num_val_samples}. Using the full validation"
            " dataset as a single batch.",
            UserWarning,
        )
    else:
        val_batch_size = batch_size

    # train
    (
        final_adam_state,
        final_model_state,
        final_model_rng,
        final_epoch,
        best_val_loss,
        best_epoch,
    ) = _train(
        batch_rng,
        update_fn,
        adam_state,
        get_params,
        update_params_direct,
        init_state,  # initial model state
        init_rng,  # initial model rng
        X,
        Y,
        Y_unc,
        X_val,
        Y_val,
        Y_val_unc,
        loss_fn,
        grad_loss_fn,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        start_epoch=0,  # always start from 0
        epochs=epochs,
        target_loss=target_loss,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        callback=callback,
        unroll=unroll,
        verbose=verbose,
    )

    # return the final parameters
    return (
        jax.tree.map(
            get_params,
            final_adam_state,
            is_leaf=lambda x: isinstance(x, OptimizerState),
        ),
        final_model_state,
        final_model_rng,
        final_epoch,
        final_adam_state,
    )


def make_loss_fn(fn_name: str, model_fn: Callable):
    """
    Create a loss function based on the model function.

    The model function should take the input data, parameters, training flag,
    state, and rng key and return the predicted output and new state.

    The loss function will return (loss, new_state), and gradients can be
    taken by passing `has_aux=True` to `jax.grad` or `jax.value_and_grad`.
    """
    if fn_name == "mse":

        def loss_fn(X, Y, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # abs(Y_pred - Y) ** 2
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs_sqr(pmm_tree_util.sub(Y_pred, Y))
                ),
                new_state,
            )

    elif fn_name == "mae":

        def loss_fn(X, Y, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # abs(Y_pred - Y)
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(pmm_tree_util.sub(Y_pred, Y))
                ),
                new_state,
            )

    elif fn_name == "mse_unc":
        # MSE with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # Y_unc is assumed to be the uncertainty in the targets
            # abs(Y_pred - Y) ** 2 / Y_unc
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs_sqr(
                        pmm_tree_util.div(pmm_tree_util.sub(Y_pred, Y), Y_unc)
                    )
                ),
                new_state,
            )

    elif fn_name == "mae_unc":
        # MAE with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # Y_unc is assumed to be the uncertainty in the targets
            # abs(Y_pred - Y) / Y_unc
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(
                        pmm_tree_util.div(pmm_tree_util.sub(Y_pred, Y), Y_unc)
                    )
                ),
                new_state,
            )

    elif fn_name == "mre":
        # Mean relative error
        def loss_fn(X, Y, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # abs((Y_pred - Y) / (Y + 1e-4))
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(
                        pmm_tree_util.div(
                            pmm_tree_util.sub(Y_pred, Y),
                            pmm_tree_util.scalar_add(Y, 1e-4),
                        )
                    )
                ),
                new_state,
            )

    elif fn_name == "mre_unc":
        # Mean relative error with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # abs((Y_pred - Y) / ((Y + 1e-4) * Y_unc))
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(
                        pmm_tree_util.div(
                            pmm_tree_util.sub(Y_pred, Y),
                            pmm_tree_util.mul(
                                pmm_tree_util.scalar_add(Y, 1e-4),
                                Y_unc,
                            ),
                        )
                    )
                ),
                new_state,
            )

    elif fn_name == "mrd":
        # mean relative difference
        def loss_fn(X, Y, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # abs((Y_pred - Y) / ((Y + Y_pred) * 2.0 + 1e-4))
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(
                        pmm_tree_util.div(
                            pmm_tree_util.sub(Y_pred, Y),
                            pmm_tree_util.scalar_add(
                                pmm_tree_util.scalar_mul(
                                    pmm_tree_util.add(Y, Y_pred),
                                    2.0,
                                ),
                                1e-4,
                            ),
                        )
                    )
                ),
                new_state,
            )

    elif fn_name == "mrd_unc":
        # mean relative difference with uncertainty in the targets
        def loss_fn(X, Y, Y_unc, params, training, state, rng):
            Y_pred, new_state = model_fn(X, params, training, state, rng)
            # abs((Y_pred - Y) / (((Y + Y_pred) * 2.0 + 1e-4) * Y_unc))
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(
                        pmm_tree_util.div(
                            pmm_tree_util.sub(Y_pred, Y),
                            pmm_tree_util.mul(
                                pmm_tree_util.scalar_add(
                                    pmm_tree_util.scalar_mul(
                                        pmm_tree_util.add(Y, Y_pred),
                                        2.0,
                                    ),
                                    1e-4,
                                ),
                                Y_unc,
                            ),
                        )
                    )
                ),
                new_state,
            )

    elif fn_name == "mse_unsupervised":

        def loss_fn(X, params, training, state, rng):
            """
            Mean squared error loss function for unsupervised training.
            """
            X_pred, new_state = model_fn(X, params, training, state, rng)
            # abs(X_pred - X) ** 2
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs_sqr(pmm_tree_util.sub(X_pred, X))
                ),
                new_state,
            )

    elif fn_name == "mae_unsupervised":

        def loss_fn(X, params, training, state, rng):
            """
            Mean absolute error loss function for unsupervised training.
            """
            X_pred, new_state = model_fn(X, params, training, state, rng)
            # abs(X_pred - X)
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(pmm_tree_util.sub(X_pred, X))
                ),
                new_state,
            )

    elif fn_name == "mre_unsupervised":

        def loss_fn(X, params, training, state, rng):
            """
            Mean relative error loss function for unsupervised training.
            """
            X_pred, new_state = model_fn(X, params, training, state, rng)
            # abs((X_pred - X) / (X + 1e-4))
            return (
                pmm_tree_util.mean(
                    pmm_tree_util.abs(
                        pmm_tree_util.div(
                            pmm_tree_util.sub(X_pred, X),
                            pmm_tree_util.scalar_add(X, 1e-4),
                        )
                    )
                ),
                new_state,
            )

    else:
        raise ValueError(f"Unknown loss function: {fn_name}")

    return loss_fn
