import flax.linen as nn
import jax
from .resnet_blocks import UpBlock
import jax.numpy as jnp


class RPeak(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = UpBlock(64, 2)(x)
        x = UpBlock(32, 2)(x)
        x = UpBlock(16, 2)(x)
        x = nn.Conv(1, kernel_size=[1], kernel_init=nn.initializers.zeros)(x)
        return x


class Analyser(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        # x = jnp.expand_dims(x, axis=-1)  # expand (B, L) to (B, L, 1)
        x = UpBlock(64, 2)(x)
        x = UpBlock(32, 2)(x)
        x = UpBlock(16, 2)(x)
        x = nn.Conv(1, kernel_size=[1], kernel_init=nn.initializers.zeros)(x)
        x = jnp.reshape(x, (-1, 1024))
        return x
