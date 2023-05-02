import flax
import jax
import optax
from model.model import DiffusionModel
import jax.numpy as jnp
from train_diffusion import TrainState


def get_ddim(rng, config):
    model = DiffusionModel(
        feature_sizes=config.feature_sizes,
        block_depths=config.block_depths,
        attention_depths=config.attention_depths,
    )
    rng_diffusion, rng_params = jax.random.split(rng)
    dummy_series = jnp.ones((1, 1024), dtype=jnp.float32)
    dummy_labels = jnp.ones((1, 3), dtype=jnp.int32)
    dummy_dict = [{"series": i, "peaks": l}
                  for i, l in zip(dummy_series, dummy_labels)]
    dummy_merged = dummy_series, dummy_labels
    variables = model.init(rng_params, rng_diffusion, dummy_merged, train=True)
    vars = flax.training.checkpoints.restore_checkpoint(
        ckpt_dir=config.ddim_path, target=None, step=29)
    # print(vars)
    tx = optax.adamw(learning_rate=0.01,
                     weight_decay=0.01)
    sample_state = TrainState.create(
        apply_fn=model.apply,
        params=vars["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
        ema_params=variables["params"],
        ema_momentum=0.01
    )

    restored_state = flax.training.checkpoints.restore_checkpoint(
        ckpt_dir=config.ddim_path, target=sample_state)
    return restored_state
