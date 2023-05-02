import flax.linen as nn
import jax
import jax.numpy as jnp

nonlinearity = nn.swish


class ResnetBlock(nn.Module):
    width: int = None
    dropout: float = 0.0
    kernel_size: int = 10
    # strides: tuple = (1,)
    # resample: Optional[str] = None

    @nn.compact
    def __call__(self, h_in, train: bool):
        residual = nn.Conv(features=self.width, kernel_size=[1])(h_in)
        h = nn.BatchNorm(use_running_average=not train,
                         use_bias=False, use_scale=False)(h_in)
        h = nn.Conv(features=self.width, kernel_size=[self.kernel_size])(h)
        h = nonlinearity(h)
        h = nn.Conv(features=self.width, kernel_size=[self.kernel_size])(h)
        return h + residual


class DownBlock(nn.Module):
    width: int = None
    block_depth: int = None
    dropout: float = 0.0

    @nn.compact
    def __call__(self, h_in, train: bool):
        B, L, C = h_in.shape
        skips = []
        # h_next = h_in
        h = h_in
        for _ in range(self.block_depth):
            # h_old = h_next
            h = ResnetBlock(width=self.width,
                            dropout=self.dropout,
                            )(h, train=train)
            skips.append(h)

        h = nn.avg_pool(h, (2,), strides=(2,))
        return [h, skips]


class UpBlock(nn.Module):
    width: int = None
    # skips: nn.Module = None
    block_depth: int = None
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, skips=None, train: bool = True):
        B, L, C = x.shape  # seperate in Batch, length, and channels
        x = jax.image.resize(x, shape=(B, L * 2, C), method="bilinear")
        for _ in range(self.block_depth):
            if skips != None:
                x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResnetBlock(width=self.width)(x, train=train)
        return x
