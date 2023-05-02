import argparse
import flax
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as onp
import optax

from model.model import DiffusionModel
from model_loader import get_ddim
from train_diffusion import TrainState
#from train import TrainState, create_train_state, DiffusionModel

SERIES_LENGTH = 1024


def get_dataset(rng):
    """Loads the Dataset and preprocesses it"""
    data = jnp.load("../data/synthetic/beats3.npy")
    labels = jnp.load("../data/synthetic/beats3_labels.npy")

    shuffled_indices = jax.random.permutation(rng, data.shape[0])
    randomized_data = data[shuffled_indices, :]
    randomized_labels = labels[shuffled_indices, :]

    # take the first 60k
    randomized_data = randomized_data[0:60032]
    randomized_labels = randomized_labels[0:60032]

    print(len(randomized_data))
    set_zero_labels = jax.random.randint(rng, [60032], 0, 2)
    print(set_zero_labels[0:10])
    print(randomized_labels[0:10])
    #randomized_labels = randomized_labels.at[::3].set([-1, -1, -1])
    print(randomized_labels[0:10])
    del data
    randomized_data = randomized_data.astype("float32")

    # series_batch = jnp.array_split(randomized_data, 60032 // 64)
    # label_batch = jnp.array_split(
    #    randomized_labels, 60032 // 64)

    merged = randomized_data, randomized_labels
    size = randomized_data.size
    itemsize = randomized_data.itemsize
    print("Dataset succesfully loaded. Size in bytes: ", size * itemsize)

    return merged


def generate_series(rng, state, amount, noise=None, peak_locations=None, variance=0.0):
    rng, noise_rng = jax.random.split(rng)
    mu = 0.0
    sigma = 0.1
    initial_noise = jax.random.normal(noise_rng, (amount, 1024, 1))
    initial_noise = sigma * initial_noise + mu

    rng, label_rng_A, label_rng_B, label_rng_C, label_rng_D = jax.random.split(
        rng, num=5)
    labelA = jax.random.randint(
        label_rng_A, shape=(amount, 1), minval=100, maxval=350)
    labelB = jax.random.randint(
        label_rng_B, shape=(amount, 1), minval=400, maxval=650)
    labelC = jax.random.randint(
        label_rng_C, shape=(amount, 1), minval=700, maxval=1020)
    labelD = jax.random.randint(label_rng_D, shape=(
        amount, 1), minval=950, maxval=1024)
    labels = jnp.concatenate([labelA, labelB, labelC], axis=1)
    labels = jnp.expand_dims(labels, axis=-1)
    # print(labels)
    # print(labels.shape)
    if peak_locations is None:
        peak_locations = labels
    if noise is None:
        noise = initial_noise

    #step_offset = 0.7 if is_real_data else 0.0

    variables = {"params": state.params, "batch_stats": state.batch_stats}
    generated_series, steps, codes = state.apply_fn(variables,
                                                    noise,
                                                    peak_locations,
                                                    10,
                                                    variance,
                                                    method=DiffusionModel.reverse_diffusion
                                                    )
    return jnp.squeeze(generated_series), peak_locations, steps


def plot_samples(set, path: str, labels=None, title=None):
    """
    Plots a 3 x 3 grid of samples drawn from the set, and saves
    it under the given path
    """
    num_rows = 3
    num_cols = 3
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)

            ax = plt.gca()
            ax.set_ylim([-0.5, 1])
            plt.plot(set[index], 'k,-')
            plt.axis("off")
            if labels != None:
                title_str = f"{labels[index]}".replace(
                    "[", "").replace("]", "").replace("\n", " ")
                plt.title(title_str, fontsize=8)
                plt.vlines(labels[index], ymin=-0.5,
                           ymax=1.0, colors="red", alpha=0.1)
    if title != None:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def downsample(series_batch, current_frequency, new_frequency):
    series = series_batch
    batch = 1
    series_length = len(series)
    if len(series_batch.shape) > 1:
        batch = series_batch[0]
        series_length = series_batch.shape[1]

    seconds = series_length / current_frequency
    new_length = int(seconds * new_frequency)

    downsampled = jax.image.resize(series, (batch, new_length))


def analyze(model_state, rng, dataset):
    #! Superresolution via inpainting
    peaks = jnp.repeat(jnp.array([-1, -1, - 1]), 3)
    peaks = jnp.reshape(peaks, (-1, 3, 1))
    input_series = dataset[0][0:3]
    labels = dataset[1][0:3]

    downsampled = jax.image.resize(input_series, (3, 128), "lanczos5")
    rng, noise_rng = jax.random.split(rng)
    repeated = jnp.repeat(downsampled, 8)
    every_eights = jnp.arange(0, 1024 * 3, 8)
    mask = jnp.zeros((3*1024), dtype=jnp.bool_)
    mask = mask.at[every_eights].set(1)
    mask = ~mask

    masked_input = input_series.reshape((-1))
    masked_input = masked_input.at[mask].set(jnp.nan)

    masked_input = masked_input.reshape((3, -1))
    mask = mask.reshape(3, -1)

    input_data = repeated
    input_data = input_data.reshape((3, -1, 1))

    # Put it through the models inpaint method

    variables = {"params": state.params, "batch_stats": state.batch_stats}
    inpaint_rng, rng = jax.random.split(rng)
    input_series = jnp.expand_dims(input_series, -1)
    labels = jnp.expand_dims(labels, -1)
    generated_series, steps, predictions = state.apply_fn(variables,
                                                          inpaint_rng,
                                                          input_data,
                                                          mask,
                                                          peaks,
                                                          29,
                                                          variance=0.1,
                                                          seed_with_series=True,
                                                          method=DiffusionModel.superresolution
                                                          )
    # return jnp.squeeze(generated_series), peak_locations, steps
    generated_series = generated_series.squeeze()

    plot_data = [
        input_series[0],
        input_series[1],
        input_series[2],

        masked_input[0],
        masked_input[1],
        masked_input[2],

        generated_series[0],
        generated_series[1],
        generated_series[2]


    ]
    plot_samples(plot_data, f"test_sr.png", [*labels, *labels, *labels],
                 title=f"Real data superresolution inpainting 64Hz -> 512Hz, no peak info")

    # plt.figure()
    num_rows = int(onp.ceil(len(predictions) / 6))
    for i in range(len(steps)):
        step = steps[i]
        series = step[0]
        plt.subplot(num_rows, 6, i + 1)
        plt.plot(series)
        ax = plt.gca()

        ax.set_ylim([-0.5, 1])

        plt.axis("off")
    plt.tight_layout()
    plt.savefig("steps_sr.png")
    plt.close()

    for i in range(len(predictions)):

        step = predictions[i]
        series = step[0]
        plt.subplot(num_rows, 6, i + 1)
        plt.plot(series)
        ax = plt.gca()

        ax.set_ylim([-0.5, 1])

        plt.axis("off")
    plt.tight_layout()
    plt.savefig("predictions_sr.png")
    plt.close()

    exit()

    #!Inpaint:
    print("Plotting restored data . . .")
    peaks = jnp.repeat(jnp.array([-1, -1, -1]), 3)
    peaks = jnp.reshape(peaks, (-1, 3, 1))

    input_series = dataset[0][0:3]

    #input_peaks = dataset[1][0:3]
    labels = dataset[1][0:3]
    print(labels)
    middle_peaks = jnp.array([labels[0][0], labels[1][2], labels[2][2]])
    print(middle_peaks)
    # exit()
    mask = jnp.zeros(3 * 1024, dtype=jnp.bool_)
    mask = mask.at[middle_peaks[0] - 50: middle_peaks[0] + 50].set(1)
    mask = mask.at[middle_peaks[1] + 1024 -
                   100: middle_peaks[1] + 1024 + 100].set(1)
    mask = mask.at[middle_peaks[2] + 2048 -
                   350: middle_peaks[2] + 2048 + 550].set(1)
    # print(mask.at[131])
    masked_input = input_series.reshape((-1))
    indice_mask = jnp.arange(0, 1024 * 3, dtype=jnp.int16)
    indice_mask = indice_mask[mask]
    print(indice_mask)
    masked_input = masked_input.at[mask].set(jnp.nan)
    masked_input = masked_input.reshape((3, -1))
    mask = mask.reshape((3, -1))

    # Put it through the models inpaint method
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    inpaint_rng, rng = jax.random.split(rng)
    input_series = jnp.expand_dims(input_series, -1)
    labels = jnp.expand_dims(labels, -1)
    generated_series, steps, predictions = state.apply_fn(variables,
                                                          inpaint_rng,
                                                          input_series,
                                                          mask,
                                                          labels,
                                                          29,
                                                          method=DiffusionModel.inpaint
                                                          )
    # return jnp.squeeze(generated_series), peak_locations, steps
    generated_series = generated_series.squeeze()

    plot_data = [
        input_series[0],
        input_series[1],
        input_series[2],

        masked_input[0],
        masked_input[1],
        masked_input[2],

        generated_series[0],
        generated_series[1],
        generated_series[2]


    ]
    plot_samples(plot_data, f"test.png", [*labels, *labels, *labels],
                 title=f"Real data inpainting")

    # plt.figure()
    for i in range(len(steps)):
        step = steps[i]
        series = step[0]
        plt.subplot(5, 6, i + 1)
        plt.plot(series)
        ax = plt.gca()

        ax.set_ylim([-0.5, 1])

        plt.axis("off")
    plt.tight_layout()
    plt.savefig("steps.png")
    plt.close()

    for i in range(len(predictions)):
        step = predictions[i]
        series = step[0]
        plt.subplot(5, 6, i + 1)
        plt.plot(series)
        ax = plt.gca()

        ax.set_ylim([-0.5, 1])

        plt.axis("off")
    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.close()
    exit()

    #! Load real data, downsample und use model as superresolution

    peaks = jnp.repeat(jnp.array([-1, -1, - 1]), 3)
    peaks = jnp.reshape(peaks, (-1, 3, 1))
    input_series = dataset[0][0:3]
    labels = dataset[1][0:3]

    downsampled = jax.image.resize(input_series, (3, 128), "lanczos5")
    rng, noise_rng = jax.random.split(rng)
    noisy = jax.random.normal(noise_rng, (3, 512))
    noisy = 0.01 * noisy
    downsampled = downsampled.reshape((-1))
    repeated = jnp.repeat(downsampled, 8)
    repeated = repeated.reshape((-1))
    noisy = noisy.reshape((-1))
    #downsampled = downsampled.reshape((-1))
    print(downsampled.shape)
    print(noisy.shape)
    mask = jnp.arange(0, 3072, 2)
    #mask = jnp.expand_dims(mask, 0)
    #mask = jnp.repeat(mask, 3, 0)
    #mask = jnp.ravel_multi_index(mask, (3, 1024))
    print(mask.shape)
    print(downsampled.shape)
    print(noisy.shape)
    print(repeated.shape)
    #repeated = repeated.at[mask].add(noisy)

    #input_data = noisy.at[mask].set(downsampled)
    input_data = repeated
    input_data = input_data.reshape((3, -1))
    input_data = jnp.expand_dims(input_data, -1)

    # Put it through the model with varying variance:

    for i in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        print(f"\tPlotting with variance {i}")
        g_rng, rng = jax.random.split(rng)
        series, peak_locations, steps = generate_series(
            g_rng, model_state, amount=3, peak_locations=peaks, noise=input_data, variance=i
        )
        steps = onp.array(steps)
        # 0, 1, 2 real data, 3, 4, 5 step 1,  6, 7, 8 step 10
        plot_data = [
            input_series[0],
            input_series[1],
            input_series[2],

            input_data[0],
            input_data[1],
            input_data[2],


            steps[10][0],
            steps[10][1],
            steps[10][2],
        ]
        plot_labels = [
            *labels,
            *labels,
            *labels,
        ]
        plot_samples(plot_data, f"upsampled_{i}.png", plot_labels,
                     title=f"Real data, partly noisy, variance {i}")

    # plot_data = [
    #     input_series[0],
    #     input_series[1],
    #     input_series[2],

    #     input_data[0],
    #     input_data[1],
    #     input_data[2],

    #     input_data[0],
    #     input_data[1],
    #     input_data[2],

    # ]
    # plot_samples(plot_data, "test.png", labels)

    # exit()
    # Load real data, delete peak and ask to restore
    print("Plotting restored data . . .")
    peaks = jnp.repeat(jnp.array([-1, -1, -1]), 3)
    peaks = jnp.reshape(peaks, (-1, 3, 1))

    input_series = dataset[0][0:3]
    #input_peaks = dataset[1][0:3]
    labels = dataset[1][0:3]
    print(labels)
    middle_peaks = jnp.array([labels[0][0], labels[1][2], labels[2][2]])
    print(middle_peaks)
    # exit()
    rng, noise_rng1, noise_rng2, noise_rng3 = jax.random.split(rng, num=4)

    noise50 = jax.random.normal(noise_rng1, (100,))
    noise100 = jax.random.normal(noise_rng2, (200,))
    noise150 = jax.random.normal(noise_rng3, (300,))

    noise50 = 0.1 * noise50
    noise100 = 0.1 * noise100
    noise150 = 0.1 * noise150

    series = jnp.reshape(input_series, (1024 * 3))
    series = series.at[middle_peaks[0] - 50: middle_peaks[0] + 50].set(noise50)
    series = series.at[middle_peaks[1] + 1024 -
                       100: middle_peaks[1] + 1024 + 100].set(noise100)
    series = series.at[middle_peaks[2] + 2048 -
                       150: middle_peaks[2] + 2048 + 150].set(noise150)

    # print(middle_peaks)
    #input_series = input_series.at[middle_peaks - 50 : middle_peaks + 50].set(noise)
    dirty_series = jnp.reshape(series, (3, 1024))
    dirty_series = jnp.expand_dims(dirty_series, axis=-1)
    input_labels = jnp.expand_dims(labels, axis=-1)

    real_rng, rng = jax.random.split(rng)

    # put it through with varying variances:
    for i in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        print(f"\tPlotting with variance {i}")
        series, peak_locations, steps = generate_series(
            real_rng, model_state, amount=3, peak_locations=input_labels, noise=dirty_series, variance=i
        )
        steps = onp.array(steps)
        # 0, 1, 2 real data, 3, 4, 5 step 1,  6, 7, 8 step 10
        plot_data = [
            input_series[0],
            input_series[1],
            input_series[2],

            dirty_series[0],
            dirty_series[1],
            dirty_series[2],


            steps[10][0],
            steps[10][1],
            steps[10][2],
        ]
        plot_labels = [
            *labels,
            *labels,
            *labels,
        ]
        plot_samples(plot_data, f"peak_reconstructed_{i}.png", plot_labels,
                     title=f"Real data, partly noisy, variance {i}")

    # plot 0 peaks
    print("Plotting 0 peaks . . .")
    peaks = jnp.repeat(jnp.array([-1, -1, -1]), 9)
    peaks = jnp.reshape(peaks, (-1, 3, 1))
    peak0_rng, rng = jax.random.split(rng)
    series, peak_locations, steps = generate_series(
        peak0_rng, model_state, amount=9, peak_locations=peaks)
    plot_samples(series, "peak0.png", peaks,
                 title="Generated data, no peak info")

    # Plot 1 peak
    print("Plotting 1 peak . . .")
    peaks = jnp.linspace(10, 1010, 9).astype(int)
    peaks = jnp.reshape(peaks, (-1, 1, 1))
    peak1_rng, rng = jax.random.split(rng)
    series, peak_locations, steps = generate_series(
        peak1_rng, model_state, amount=9, peak_locations=peaks)
    plot_samples(series, "peak1.png", peaks,
                 title="Generated data, one peak provided")

    # Plot 2 peaks
    print("Plotting 2 peaks . . .")
    peakA = jnp.linspace(10, 490, 9).astype(int)
    peakB = jnp.flip(jnp.linspace(520, 1010, 9).astype(int))
    peaks = jnp.reshape(jnp.stack([peakA, peakB], axis=1), (-1, 2, 1))
    peak2_rng, rng = jax.random.split(rng)
    series, peak_locations, steps = generate_series(
        peak2_rng, model_state, amount=9, peak_locations=peaks)
    plot_samples(series, "peak2.png", peaks,
                 title="Generated data, two peaks provided")

    # plot 3 peaks
    print("Plotting 3 peaks . . .")
    peak3_rng, rng = jax.random.split(rng)
    series, peak_locations, steps = generate_series(
        peak3_rng, model_state, amount=9)
    plot_samples(series, "peak3.png", peak_locations,
                 title="Generated data, 3 peaks provided")

    # Load real data and put it through
    print("Plotting real data . . .")
    peaks = jnp.repeat(jnp.array([-1, -1, -1]), 3)
    peaks = jnp.reshape(peaks, (-1, 3, 1))

    input_series = dataset[0][0:3]
    input_series = jnp.expand_dims(input_series, axis=-1)
    labels = dataset[1][0:3]
    real_rng, rng = jax.random.split(rng)

    # put it through with varying variances:
    for i in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        print(f"\tPlotting with variance {i}")
        series, peak_locations, steps = generate_series(
            real_rng, model_state, amount=3, peak_locations=peaks, noise=input_series, variance=i
        )
        steps = onp.array(steps)
        # 0, 1, 2 real data, 3, 4, 5 step 1,  6, 7, 8 step 10
        plot_data = [
            input_series[0],
            input_series[1],
            input_series[2],

            steps[1][0],
            steps[1][1],
            steps[1][2],
            steps[10][0],
            steps[10][1],
            steps[10][2],
        ]
        plot_labels = [
            *labels,
            *labels,
            *labels,
        ]
        plot_samples(plot_data, f"peak_real_{i}.png", plot_labels,
                     title=f"Real data, variance {i}, no peak info")

    # Add noise and put it through with variable variance:
    noise_rng, rng = jax.random.split(rng)
    input_series = dataset[0][0:3]
    noises = jax.random.normal(noise_rng, (3, 1024))
    noises = noises * 0.1
    dirty_series = input_series + noises
    dirty_series = jnp.expand_dims(dirty_series, axis=-1)

    print("Plotting dirty data")
    for i in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        print(f"\tPlotting with variance {i}")
        series, peak_locations, steps = generate_series(
            real_rng, model_state, amount=3, peak_locations=peaks, noise=dirty_series, variance=i
        )
        steps = onp.array(steps)
        # 0, 1, 2 real data, 3, 4, 5 step 1,  6, 7, 8 step 10
        plot_data = [
            input_series[0],
            input_series[1],
            input_series[2],

            dirty_series[0],
            dirty_series[1],
            dirty_series[2],

            steps[10][0],
            steps[10][1],
            steps[10][2],
        ]
        plot_labels = [
            *labels,
            *labels,
            *labels,
        ]
        plot_samples(plot_data, f"peak_real_dirty_{i}.png", plot_labels,
                     title=f"Dirty real data, variance {i}, no peak info")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-fs', '--feature-sizes', nargs="+",
                        type=int, default=[32, 64, 96, 128, 160])
    # The amount of ResnetBlocks that each layer of the UNet should have
    parser.add_argument('-bd', '--block-depths', type=int, default=3)
    # The index of the UNet layer at which to start using attention. Set bigger than the UNet depths
    # to not use Attention at all.
    parser.add_argument('-ad', '--attention-depths', type=int, default=2)

    parser.add_argument('-dp', '--ddim-path', type=str, default="None")

    args = parser.parse_args()
    rng = jax.random.PRNGKey(0)
    state = get_ddim(rng, args)
    data_rng, rng = jax.random.split(rng)
    dataset = get_dataset(data_rng)

    analyze(state, rng, dataset)
