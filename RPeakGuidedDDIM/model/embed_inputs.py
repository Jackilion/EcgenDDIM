import flax.linen as nn
import jax
import jax.numpy as jnp
import math


class SinEmbed(nn.Module):
    """
    Embeds an input through Sin and Cos.
    Outputs a Tensor of Shape (BatchSize, 1, EmbedDims)
    """
    embedding_dims: int = 32
    embedding_max_frequency: float = 1000.0
    embedding_min_frequency: float = 1.0

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dims // 2
            )
        )

        angular_speeds = 2.0 * math.pi * frequencies
        angular_speeds = jnp.expand_dims(angular_speeds, 0)

        embeddings = jnp.concatenate(
            [
                jnp.sin(angular_speeds * x),
                jnp.cos(angular_speeds * x)
            ],
            axis=2
        )
        return embeddings


class EmbedPeakLocation(nn.Module):
    '''
    Encodes the R peak location input of shape [p1, p2, p3]
    into a delta series of Length series_length that has it's delta peaks
    at the corresponding locations.
    '''
    series_length: int = 1024

    @nn.compact
    def __call__(self, x):

        # input is of shape (B, 3, 1)
        # reshape to (B, 3)
        B, L, C = x.shape

        x = x.reshape(B, L)
        # print(x[0])
        # if x[0] == x[1] == x[2] == -1:
        #     return jnp.zeros((B, self.series_length, 1))

        zeros = jnp.zeros(B * self.series_length)
        # if L == 0:
        #     return jnp.zeros((B, self.series_length, 1))

        non_zero_multiidx = jnp.array(
            (jnp.repeat(
                jnp.arange(B),
                L
            ),
                jnp.reshape(x, (-1,))))

        non_zero_inds = jnp.ravel_multi_index(
            non_zero_multiidx, (B, self.series_length), mode='clip')
        delta_series = jnp.reshape(
            zeros.at[non_zero_inds].set(1.0), (-1, self.series_length))

        #minus_ones_indx = x.where()
        delta_series = delta_series.at[:, 0].set(0.0)

        return delta_series


class EmbedInputs(nn.Module):
    embedding_dims_variance: int = 32

    @nn.compact
    def __call__(self, series, peaks, variance):
        #series, peaks, variance = input
        B, L, C = series.shape

        variance_embed = SinEmbed(
            embedding_dims=self.embedding_dims_variance)(variance)

        variance_embed = jnp.repeat(variance_embed, L, axis=1)

        assert variance_embed.shape == (B, L, self.embedding_dims_variance)

        peak_embed = EmbedPeakLocation()(peaks)
        peak_embed = jnp.expand_dims(peak_embed, axis=-1)

        series = nn.Conv(self.embedding_dims_variance -
                         1, kernel_size=[1])(series)
        assert series.shape == (B, L, self.embedding_dims_variance - 1)

        x = jnp.concatenate([series, peak_embed, variance_embed], axis=2)
        return x
