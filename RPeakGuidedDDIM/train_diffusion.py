from argparse import Namespace
import copy
from functools import partial
import time
from model.model import DiffusionModel
import jax
import jax.numpy as jnp
import optax
from typing import Any
from tensorboard.plugins.hparams import api as hp
from flax.training import (train_state, checkpoints)
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


SERIES_LENGTH = 1024
PLOT_IMAGE_COUNT = 18


class TrainState(train_state.TrainState):
    batch_stats: Any
    ema_params: Any = None
    ema_momentum: float = None
    epoch: int = None


def create_train_state(rng, config):
    """Creates initial TrainState to hold params"""

    model = DiffusionModel(
        feature_sizes=config.feature_sizes,
        block_depths=config.block_depths,
        attention_depths=config.attention_depths
    )
    rng_diffusion, rng_params = jax.random.split(rng)

    dummy_series = jnp.ones((1, SERIES_LENGTH), dtype=jnp.float32)
    dummy_labels = jnp.ones((1, 3), dtype=jnp.int32)
    dummy_dict = [{"series": i, "peaks": l}
                  for i, l in zip(dummy_series, dummy_labels)]
    dummy_merged = dummy_series, dummy_labels
    variables = model.init(rng_params, rng_diffusion, dummy_merged, train=True)

    tx = optax.adamw(learning_rate=config.learning_rate,
                     weight_decay=config.weight_decay)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        ema_params=variables["params"],
        ema_momentum=config.ema_momentum
    )


def plot_samples(set, path: str, labels=None, title=None):
    """
    Plots a 6 x 3 grid of samples drawn from the set, and saves
    it under the given path
    """
    num_rows = 3
    num_cols = 6
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)

            ax = plt.gca()
            ax.set_ylim([-0.5, 1])
            plt.plot(set[index])
            plt.axis("off")
            if labels != None:
                #title_str = f"{labels[index]}"
                #plt.title(title_str, fontsize=8)
                plt.vlines(labels[index], ymin=-0.5,
                           ymax=1.0, colors="red", alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_ema_params(ema_params, current_params, ema_momentum):
    return ema_momentum * ema_params + (1-ema_momentum)*current_params


@partial(jax.jit, static_argnums=3)
def train_step(rng, state, batch, loss_function: str):
    """
    The actual train step. Gets called with a batch of series and a model state.
    It processes the batch, then updates the train states parameters and returns
    the new state.
    """
    def compute_loss(params):
        outputs, mutated_vars = state.apply_fn(
            {"params": params,
             "batch_stats": state.batch_stats},
            rng, batch, train=True, mutable=["batch_stats"]
        )
        if loss_function == "mean_abs_error":
            loss_fn = mean_abs_error
        elif loss_function == "mean_sqr_error":
            loss_fn = mean_sqr_error
        noises, input_batch, pred_noises, pred_series, latent_space = outputs
        noise_loss = loss_fn(pred_noises, noises).mean()
        return noise_loss, mutated_vars

    assert batch[0].dtype in [jnp.float32, jnp.float64]

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, mutated_vars), grads = grad_fn(state.params)

    metrics = {"loss": loss,
               # "loss_ema": loss_ema
               }
    new_state = state.apply_gradients(
        grads=grads, batch_stats=mutated_vars['batch_stats'])
    # new_ema_params = jax.tree_map(
    #    compute_ema_params, state.ema_params, state.params, state.ema_momentum)
    # new_state = new_state.replace(ema_params=new_ema_params)
    return new_state, metrics


def copy_params_to_ema(state):
    state = state.replace(params_ema=state.params)
    return state


def apply_ema_decay(state, ema_decay):
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay +
                              p * (1. - ema_decay), state.params_ema, state.params)
    state = state.replace(params_ema=params_ema)
    return state


def mean_abs_error(predictions, targets):
    """
    The loss function used to compare the predicted noises
    vs the actual noises.
    """
    return jnp.abs(predictions - targets)


def mean_sqr_error(predictions, targets):
    """
    The loss function used to compare the predicted noises
    vs the actual noises.
    """
    return jnp.square(predictions - targets)


def evaluate(state, rng, diffusion_steps: int, config: Namespace):
    """
    This function produces samples using the passed model state and
    an rng key for generating the initial noise. It passes the samples
    diffusion_steps times through the network.
    """
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    generated_series, labels, steps = state.apply_fn(variables,
                                                     rng,
                                                     (18, 1024, 1),
                                                     diffusion_steps,
                                                     method=DiffusionModel.generate
                                                     )
    generated_series = jnp.squeeze(generated_series)
    # plot generates samples
    plot_samples(generated_series, config.img_dir +
                 f"/epoch_{state.epoch}", labels=labels, title=f"R-Peak guided DDIM Output after {state.epoch} epochs")
    # plot the denoising steps
    for i in range(len(steps)):
        step = steps[i]
        series = step[0]
        plt.subplot(5, 6, i + 1)
        plt.plot(series)
        ax = plt.gca()

        ax.set_ylim([-0.5, 1])

        plt.axis("off")

    plt.suptitle(f"Reverse diffusion process after {state.epoch} epochs")
    plt.tight_layout()
    name = f"/epoch_{state.epoch}_process"
    plt.savefig(config.img_dir + name)
    plt.close()


def train(dataset, config: Namespace, rng) -> TrainState:
    """
    The main starting point that manages the training
    of the model. It returns the final state after Training
    is complete.
    """
    # Hide GPU from TF, so it doesn't allocate VRAM
    tf.config.experimental.set_visible_devices([], 'GPU')

    # Init the summary writer and log all of the hyperparameters from the config
    summary_writer = tf.summary.create_file_writer(config.log_dir)
    hyper_params = {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate
    }
    with summary_writer.as_default():
        hparam_dict = copy.deepcopy(vars(config))
        hparam_dict["feature_sizes"] = ", ".join(
            str(x) for x in hparam_dict["feature_sizes"])
        ret = hp.hparams(hparam_dict)
        print(ret)
    #     tf.summary.text("epochs", str(config.epochs), step=0)
    #     tf.summary.text("learning_rate", str(config.learning_rate), step=0)
    #     tf.summary.text("weight_decay", str(config.weight_decay), step=0)
    #     tf.summary.text("batch_size", str(config.batch_size), step=0)

    series_iter, label_iter = dataset

    rng, state_rng = jax.random.split(rng)
    state = create_train_state(state_rng, config)

    train_metrics_last_t = time.time()
    for epoch in range(config.epochs):
        pbar = tqdm(range(len(series_iter)), desc=f'Epoch {epoch}')
        losses = []
        for i in pbar:
            series_batch = series_iter[i]
            label_batch = label_iter[i]

            # print(f"type of type batch: ", type(batch))
            rng, train_step_rng = jax.random.split(rng)
            state, metrics = train_step(
                rng=train_step_rng, state=state, batch=(series_batch, label_batch), loss_function=config.loss_function)
            loss = metrics["loss"]
            pbar.set_postfix({"loss": f"{loss:.5f}"})
            losses.append(metrics["loss"])

        state = state.replace(epoch=epoch)
        summary = metrics
        summary["time/seconds_per_step"] = (
            time.time() - train_metrics_last_t)
        # print("SUMMARY:::")
        # print(summary)
        evaluate(
            state, rng, diffusion_steps=config.plot_diffusion_steps, config=config)
        with summary_writer.as_default():
            tf.summary.scalar("loss", summary["loss"], epoch)
            tf.summary.scalar("time/seconds_per_epoch:",
                              summary["time/seconds_per_step"], epoch)
        train_metrics_last_t = time.time()
        # Save model
        checkpoints.save_checkpoint(
            ckpt_dir=config.ckpt_dir, target=state, step=epoch)

    # # Save model
    # checkpoints.save_checkpoint(
    #     ckpt_dir=config.chkp_dir, target=state, step=config.epochs)
    return state, rng
