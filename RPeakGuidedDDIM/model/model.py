from dataclasses import field
from typing import List
import jax
import jax.numpy as jnp
import flax.linen as nn
from .unet import UNet
# from train import BATCH_SIZE, SERIES_LENGTH


class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class InputNormalizer(nn.Module):
    normalization_method: str = "Identity"

    def setup(self):
        if self.normalization_method == "Identity":
            self.normalizer = Identity()
        elif self.normalization_method == "BatchNorm":
            self.normalizer = nn.BatchNorm(use_bias=False, use_scale=False)
        else:
            raise Exception("Unknown normalization method")

    def __call__(self, x, train: bool, **kwargs):
        if self.normalization_method == "Identity":
            return self.normalizer(x)
        elif self.normalization_method == "BatchNorm":
            return self.normalizer(x, use_running_average=not train)
        else:
            raise Exception("Unknown normalization method")

    def denormalize(self, batch):
        if self.normalization_method == "Identity":
            return batch
        elif self.normalization_method == "BatchNorm":
            norm_stats = self.normalizer.variables['batch_stats']
            # mean = norm_stats['mean']
            # print("mean shape:")  # 1024,
            # print(mean.shape)
            # print("x shape:")  # 18, 1024, 1
            # print(x.shape)

            mean = norm_stats['mean'].reshape((1, -1, 1)).astype(batch.dtype)
            var = norm_stats['var'].reshape((1, -1, 1)).astype(batch.dtype)
            std = jnp.sqrt(var + self.normalizer.epsilon)

            return std * batch + mean

        else:
            raise Exception("Unknown normalization method")


class DiffusionModel(nn.Module):
    feature_sizes: List[int] = field(
        default_factory=lambda: [32, 64, 96, 128])

    attention_depths: int = 3
    block_depths: int = 2
    # min_signal_rate: float = 0.02
    # max_signal_rate: float = 0.95
    start_log_snr: float = 2.5
    end_log_snr: float = -7.5
    schedule_type: str = "cosine"

    normalization_method: str = "Identity"

    # def normalization(x, ):

    def setup(self):

        self.normalizer = InputNormalizer(
            normalization_method=self.normalization_method)
        self.network = UNet(
            widths=self.feature_sizes,
            block_depth=self.block_depths,
            attention_depths=self.attention_depths
        )

    def __call__(self, rng, batch, train: bool):
        # print(type(batch))
        series_batch, label_batch = batch
        series_batch = self.normalizer(series_batch, train)

        #label_batch = batch["labels"]

        series_batch = jnp.expand_dims(series_batch, axis=-1)
        label_batch = jnp.expand_dims(label_batch, axis=-1)
        B, L, C = series_batch.shape
        rng, t_rng = jax.random.split(rng)
        diffusion_times = jax.random.uniform(
            t_rng, (B, 1, 1), dtype=series_batch.dtype)

        rng, n_rng = jax.random.split(rng)
        mu = 0
        sigma = 0.1
        noises = jax.random.normal(n_rng, (B, L, C), dtype=series_batch.dtype)
        noises = noises * sigma + mu

        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_series = signal_rates * series_batch
        noisy_series = noisy_series + noise_rates * noises
        # noisy_series = jnp.expand_dims(noisy_series, 2)
        # noise_variance = noise_rates ** 2
        # noise_variance = jnp.expand_dims(noise_variance, 2)

        # Go from (batch_size, series_length) to (batch_size, series_length, 1)
        # noisy_series = jnp.expand_dims(noisy_series, axis=-1)

        pred_noises, pred_series, latent_space = self.denoise(
            noisy_series,
            noise_rates,
            signal_rates,
            label_batch,
            train=train
        )
        return noises, batch, pred_noises, pred_series, latent_space

    def diffusion_schedule(self, diffusion_times):
        start_snr = jnp.exp(2.5)
        end_snr = jnp.exp(-7.5)

        start_noise_power = 1.0 / (1.0 + start_snr)
        end_noise_power = 1.0 / (1.0 + end_snr)

        if self.schedule_type == "linear":
            noise_powers = start_noise_power + diffusion_times * (
                end_noise_power - start_noise_power
            )

        elif self.schedule_type == "cosine":
            start_angle = jnp.arcsin(start_noise_power ** 0.5)
            end_angle = jnp.arcsin(end_noise_power ** 0.5)
            diffusion_angles = start_angle + \
                diffusion_times * (end_angle - start_angle)

            noise_powers = jnp.sin(diffusion_angles) ** 2

        elif self.schedule_type == "log-snr-linear":
            noise_powers = start_snr ** diffusion_times / (
                start_snr * end_snr**diffusion_times + start_snr ** diffusion_times
            )

        else:
            raise NotImplementedError("Unsupported sampling schedule")

        # signal + noise = 1
        signal_powers = 1.0 - noise_powers

        signal_rates = signal_powers ** 0.5
        noise_rates = noise_powers ** 0.5

        return noise_rates, signal_rates

    def denoise(self, noisy_series,  noise_rates, signal_rates, labels, *, train: bool):
        batch = {
            "series": noisy_series,
            "variance": noise_rates ** 2,
            "labels": labels
        }
        pred_noises, latent_space = self.network(batch, train=train)
        pred_series = (noisy_series - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_series, latent_space

        if training:
            params = state.params
        else:
            params = state.params
        variables = {'params': params, 'batch_stats': state.batch_stats}
        # pred_noises, new_model_state = state.apply_fn(
        #     variables,
        #     batch={
        #         "noise": noisy_series,
        #         "variance": noise_rates ** 2
        #     },
        #     train=training,
        #     mutable=["batch_stats"]
        # )
        # jnp.
        # print(
        #    f"TYPES: {type(noisy_series)}, {type(noise_rates)}, {type(pred_noises)}, {type(signal_rates)}")
        # print(pred_noises)
        pred_series = (noisy_series - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_series

    def reverse_diffusion(self, initial_noise, labels, steps, step_offset=0.0):
        """
        Takes noise (or series) as an input, as well as rpeak locations.
        Calls the model repeatedly, then returns the final result, all steps,
        as well as the latent space of the final pass.
        """
        num_series = initial_noise.shape[0]
        step_size = (1.0 - step_offset) / steps

        all_denoising_steps = []
        all_denoising_steps.append(initial_noise)

        next_noisy_series = initial_noise
        for step in range(steps):
            noisy_series = next_noisy_series
            diffusion_times = jnp.ones(
                (num_series, 1, 1), dtype=initial_noise.dtype) - step * step_size - step_offset
            noise_rates, signal_rates = self.diffusion_schedule(
                diffusion_times)
            pred_noises, pred_series, latent_space = self.denoise(
                noisy_series, noise_rates, signal_rates, labels, train=False)
            all_denoising_steps.append(pred_series)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times)
            next_noisy_series = (next_signal_rates *
                                 pred_series + next_noise_rates * pred_noises)
        return pred_series, all_denoising_steps, latent_space

    def denormalize(self, x):
        return self.normalizer.denormalize(x)

    # def produce_latent_space_vmap(self, series, diffusion_steps: int = 1):
    #     """Used to encode a huge amount of data in parallel"""

    def produce_latent_space(self, series, diffusion_steps: int = 5):
        #num_series = series.shape[0]
        # step_size = 1.0 / diffusion_steps
        # Need to expand dims to get it to (B, L, 1), instead of (B, L)
        series = jnp.expand_dims(series, axis=-1)
        batch_size = series.shape[0]
        labels = jnp.repeat(jnp.array([-1, -1, -1]), batch_size)
        labels = jnp.reshape(labels, (-1, 3, 1))
        # Call the model, but tell it it's already almost noiseless.
        generated_series, steps, latent_space = self.reverse_diffusion(
            series, labels=labels, steps=diffusion_steps, step_offset=0.9)
        return latent_space

    # def probabilistic_sampling(self,)

    def superresolution(self, rng, series, mask, labels, steps: int = 50, variance=0.0, seed_with_series=False):
        rng, noise_rng = jax.random.split(rng)
        initial_noise = jax.random.normal(noise_rng, series.shape)
        initial_noise = 0.1 * initial_noise

        mask = mask.reshape((-1))

        #flat_series = series.reshape((-1))
        #initial_noise = initial_noise.reshape((-1))
        #initial_noise = initial_noise.at[~mask].set(flat_series[~mask])

        #initial_noise = initial_noise.reshape(series.shape)

        # inverted_mask = mask.at[]

        B, L, C = series.shape
        step_size = (1.0 - variance) / steps
        num_series = B
        all_denoising_steps = []
        all_denoising_steps.append(initial_noise)
        all_denoising_predictions = []

        if seed_with_series:
            next_noisy_series = series
        else:
            next_noisy_series = initial_noise
        for step in range(steps):
            noisy_series = next_noisy_series
            diffusion_times = jnp.ones(
                (num_series, 1, 1), dtype=initial_noise.dtype
            ) - step * step_size - variance
            noise_rates, signal_rates = self.diffusion_schedule(
                diffusion_times)
            pred_noises, pred_series, latent_space = self.denoise(
                noisy_series, noise_rates, signal_rates, labels, train=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            # Calculate the next input:
            # 1.

            epsilon_rng, rng = jax.random.split(rng)
            noises = jax.random.normal(epsilon_rng, series.shape)
            noises = 0.001 * (steps - (step + 1)) * noises
            pred_noises = pred_noises + noises

            next_noisy_series = (next_signal_rates *
                                 pred_series + next_noise_rates * pred_noises)
            weighted_original_series = (
                next_signal_rates * series + next_noise_rates * initial_noise)
            # Flatten both
            next_noisy_series = jnp.reshape(next_noisy_series, (-1))
            weighted_original_series = jnp.reshape(
                weighted_original_series, (-1))

            # Write the (noisy) points we want to keep into the next input
            # inverted_mask =
            next_noisy_series = next_noisy_series.at[~mask].set(
                weighted_original_series[~mask])

            # reshape back
            next_noisy_series = jnp.reshape(next_noisy_series, (B, -1, 1))
            all_denoising_steps.append(next_noisy_series)
            all_denoising_predictions.append(pred_series)

        return pred_series, all_denoising_steps, all_denoising_predictions

    def inpaint(self, rng, series, mask, labels, steps: int = 50, variance=0.0, seed_with_series=False):
        rng, noise_rng = jax.random.split(rng)
        initial_noise = jax.random.normal(noise_rng, series.shape)
        initial_noise = 0.1 * initial_noise

        mask = mask.reshape((-1))

        #flat_series = series.reshape((-1))
        #initial_noise = initial_noise.reshape((-1))
        #initial_noise = initial_noise.at[~mask].set(flat_series[~mask])

        #initial_noise = initial_noise.reshape(series.shape)

        # inverted_mask = mask.at[]

        B, L, C = series.shape
        step_size = (1.0 - variance) / steps
        num_series = B
        all_denoising_steps = []
        all_denoising_steps.append(initial_noise)
        all_denoising_predictions = []

        if seed_with_series:
            next_noisy_series = series
        else:
            next_noisy_series = initial_noise
        for step in range(steps):
            noisy_series = next_noisy_series
            diffusion_times = jnp.ones(
                (num_series, 1, 1), dtype=initial_noise.dtype
            ) - step * step_size - variance
            noise_rates, signal_rates = self.diffusion_schedule(
                diffusion_times)
            pred_noises, pred_series, latent_space = self.denoise(
                noisy_series, noise_rates, signal_rates, labels, train=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            # Calculate the next input:
            # 1.

            next_noisy_series = (next_signal_rates *
                                 pred_series + next_noise_rates * pred_noises)
            epsilon_rng, rng = jax.random.split(rng)
            noises = jax.random.normal(epsilon_rng, series.shape)
            noises = 0.01 * noises
            #pred_noises = pred_noises + noises
            weighted_original_series = (
                next_signal_rates * series + next_noise_rates * initial_noise)
            # Flatten both
            next_noisy_series = jnp.reshape(next_noisy_series, (-1))
            weighted_original_series = jnp.reshape(
                weighted_original_series, (-1))

            # Write the (noisy) points we want to keep into the next input
            # inverted_mask =
            next_noisy_series = next_noisy_series.at[~mask].set(
                weighted_original_series[~mask])

            # reshape back
            next_noisy_series = jnp.reshape(next_noisy_series, (B, -1, 1))
            all_denoising_steps.append(next_noisy_series)
            all_denoising_predictions.append(pred_series)

        return pred_series, all_denoising_steps, all_denoising_predictions

    def generate(self, rng, series_shape, diffusion_steps: int):
        rng, noise_rng = jax.random.split(rng)
        mu = 0.0
        sigma = 0.1
        initial_noise = jax.random.normal(noise_rng, series_shape)
        initial_noise = sigma * initial_noise + mu

        B, L, C = series_shape

        rng, label_rng_A, label_rng_B, label_rng_C = jax.random.split(
            rng, num=4)
        labelA = jax.random.randint(
            label_rng_A, shape=(B, 1), minval=100, maxval=350)
        labelB = jax.random.randint(
            label_rng_B, shape=(B, 1), minval=400, maxval=650)
        labelC = jax.random.randint(
            label_rng_C, shape=(B, 1), minval=700, maxval=950)
        labels = jnp.concatenate([labelA, labelB, labelC], axis=1)
        labels = jnp.expand_dims(labels, axis=-1)
        # zero_labels = jnp.zeros_like(labels)  # ! TODO: REMOVE FOR RPEAK GUIDED
        # labels =
        generated_series, steps, latent_space = self.reverse_diffusion(
            initial_noise, labels, diffusion_steps)

        denormalized = self.denormalize(generated_series)

        denormalized_steps = []
        for step in steps:
            denormalized_steps.append(
                self.denormalize(step))
        return denormalized, labels, steps
