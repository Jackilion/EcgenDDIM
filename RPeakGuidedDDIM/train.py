from argparse import Namespace

import jax
import jax.numpy as jnp
from model_loader import get_ddim
import train_diffusion
import train_analyser


def get_dataset(rng, config):
    """Loads the Dataset and preprocesses it"""
    data = jnp.load("../data/synthetic/beats3.npy")
    labels = jnp.load("../data/synthetic/beats3_labels.npy")

    shuffled_indices = jax.random.permutation(rng, data.shape[0])

    randomized_data = data[shuffled_indices, :]
    randomized_labels = labels[shuffled_indices, :]

    # take the first 60k
    randomized_data = randomized_data[0:60032]
    randomized_labels = randomized_labels[0:60032]

    set_zero_labels = jax.random.randint(rng, [60032], 0, 2)

    del data
    randomized_data = randomized_data.astype("float32")

    series_batch = jnp.array_split(randomized_data, 60032 // config.batch_size)
    label_batch = jnp.array_split(
        randomized_labels, 60032 // config.batch_size)

    merged = series_batch, label_batch

    size = randomized_data.size
    itemsize = randomized_data.itemsize
    print("Dataset succesfully loaded. Size in bytes: ", size * itemsize)


    return merged


def train(config: Namespace):
    """
    The main train loop. Goes through the training data set and calls train_step for each batch.
    """
    rng = jax.random.PRNGKey(config.seed)
    d_rng, rng = jax.random.split(rng)
    dataset = get_dataset(d_rng, config)  # form: (series_iter, peak_iter)

    if config.train_ddim:
        # train the diffusion model to completion:
        print("[INFO]: Starting training of DDIM.")
        ddim, rng = train_diffusion.train(dataset, config, rng)
    else:
        print("[INFO]: Loading pretrained DDIM.")
        ddim_rng, rng = jax.random.split(rng)
        ddim = get_ddim(ddim_rng, config)
