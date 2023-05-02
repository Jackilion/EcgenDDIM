from typing import List

from model.analyser import Analyser
from .resnet_blocks import ResnetBlock, DownBlock, UpBlock
from .embed_inputs import EmbedInputs
import flax.linen as nn
import jax
import jax.numpy as jnp


class Encoder(nn.Module):
    block_depths: int = None
    widths: List = None
    attention_depths: int = 3

    @nn.compact
    def __call__(self, x, train: bool = True):
        skips = []
        for index, width in enumerate(self.widths[:-1]):
            x, skip = DownBlock(
                width=width,
                block_depth=self.block_depths
            )(x, train=train)
            if index > self.attention_depths and self.attention_depths < len(self.widths):
                x = nn.SelfAttention(num_heads=4)(x)
            skips.append(skip)
        for _ in range(self.block_depths):
            x = ResnetBlock(self.widths[-1])(x, train=train)
        if len(self.widths) > self.attention_depths:
            x = nn.SelfAttention(num_heads=4)(x)
        return x, skips


class Decoder(nn.Module):
    block_depths: int = None
    widths: List = None
    attention_depths: int = 3
    #skips: List = None

    @nn.compact
    def __call__(self, x, skips, train: bool = True):
        for index, width in enumerate(reversed(self.widths[:-1])):
            skip = skips.pop()
            x = UpBlock(width, self.block_depths)(x, skip, train=train)
            if index < self.attention_depths and self.attention_depths < len(self.widths):
                x = nn.SelfAttention(num_heads=4)(x)

        x = nn.Conv(1, kernel_size=[1], kernel_init=nn.initializers.zeros)(x)
        return x


class UNet(nn.Module):
    embedding_dims: int = 32
    attention_depths: int = 3
    widths: List = None
    block_depth: int = None

    def setup(self):
        self.input_embedder = EmbedInputs(
            embedding_dims_variance=self.embedding_dims)
        self.encoder = Encoder(block_depths=self.block_depth,
                               attention_depths=self.attention_depths, widths=self.widths)
        self.decoder = Decoder(block_depths=self.block_depth,
                               attention_depths=self.attention_depths, widths=self.widths)
        self.conv1 = nn.Conv(self.embedding_dims - 1, kernel_size=[1])

    def __call__(self, batch: dict[str, jnp.ndarray], train: bool = True):
        B, L, C = batch["series"].shape

        inputs = self.input_embedder(
            batch["series"], batch["labels"], batch["variance"])

        latent_space, skips = self.encoder(inputs, train=train)
        predicted_series = self.decoder(latent_space, skips, train=train)
        del skips

        assert predicted_series.shape == (B, L, 1)
        return predicted_series, latent_space
