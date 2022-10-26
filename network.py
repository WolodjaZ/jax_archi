from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class InceptionBlock(nn.Module):
    c_red : dict  # Dictionary of reduced dimensionalities with keys "1x1", "3x3", "5x5", and "max"
    c_out : dict  # Dictionary of output feature sizes with keys "1x1", "3x3", "5x5", and "max"
    act_fn : callable   # Activation function
    init_kernel: Callable # Kernel initializer
    batch_norm: bool = True # Whether to use batch normalization

    @nn.compact
    def __call__(self, x, train=True):
        # 1x1 convolution branch
        x_1x1 = nn.Conv(self.c_out["1x1"], kernel_size=(1, 1), kernel_init=self.init_kernel, use_bias=False)(x)
        if self.batch_norm:
            x_1x1 = nn.BatchNorm()(x_1x1, use_running_average=not train)
        else:
            x_1x1 = nn.LayerNorm()(x_1x1)
        x_1x1 = self.act_fn(x_1x1)

        # 3x3 convolution branch
        x_3x3 = nn.Conv(self.c_red["3x3"], kernel_size=(1, 1), kernel_init=self.init_kernel, use_bias=False)(x)
        if self.batch_norm:
            x_3x3 = nn.BatchNorm()(x_3x3, use_running_average=not train)
        else:
            x_3x3 = nn.LayerNorm()(x_3x3)
        x_3x3 = self.act_fn(x_3x3)
        x_3x3 = nn.Conv(self.c_out["3x3"], kernel_size=(3, 3), kernel_init=self.init_kernel, use_bias=False)(x_3x3)
        if self.batch_norm:
            x_3x3 = nn.BatchNorm()(x_3x3, use_running_average=not train)
        else:
            x_3x3 = nn.LayerNorm()(x_3x3)
        x_3x3 = self.act_fn(x_3x3)

        # 5x5 convolution branch
        x_5x5 = nn.Conv(self.c_red["5x5"], kernel_size=(1, 1), kernel_init=self.init_kernel, use_bias=False)(x)
        if self.batch_norm:
            x_5x5 = nn.BatchNorm()(x_5x5, use_running_average=not train)
        else:
            x_5x5 = nn.LayerNorm()(x_5x5)
        x_5x5 = self.act_fn(x_5x5)
        x_5x5 = nn.Conv(self.c_out["5x5"], kernel_size=(5, 5), kernel_init=self.init_kernel, use_bias=False)(x_5x5)
        if self.batch_norm:
            x_5x5 = nn.BatchNorm()(x_5x5, use_running_average=not train)
        else:
            x_5x5 = nn.LayerNorm()(x_5x5)
        x_5x5 = self.act_fn(x_5x5)

        # Max-pool branch
        x_max = nn.max_pool(x, (3, 3), strides=(2, 2))
        x_max = nn.Conv(self.c_out["max"], kernel_size=(1, 1), kernel_init=self.init_kernel, use_bias=False)(x)
        if self.batch_norm:
            x_max = nn.BatchNorm()(x_max, use_running_average=not train)
        else:
            x_max = nn.LayerNorm()(x_max)
        x_max = self.act_fn(x_max)

        x_out = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_max], axis=-1)
        return x_out


class GoogleNet(nn.Module):
    num_classes : int
    act_fn : Callable
    init_kernel: Callable = nn.initializers.kaiming_normal()
    batch_norm: bool = True
    

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(64, kernel_size=(3, 3), kernel_init=self.init_kernel, use_bias=False)(x)
        if self.batch_norm:
            x = nn.BatchNorm()(x, use_running_average=not train)
        else:
            x = nn.LayerNorm()(x)
        x = self.act_fn(x)

        # Stacking inception blocks
        inception_blocks = [
            InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
            InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
            lambda inp: nn.max_pool(inp, (3, 3), strides=(2, 2)),  # 32x32 => 16x16
            InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
            InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
            InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
            InceptionBlock(c_red={"3x3": 32, "5x5": 16}, c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
            lambda inp: nn.max_pool(inp, (3, 3), strides=(2, 2)),  # 16x16 => 8x8
            InceptionBlock(c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
            InceptionBlock(c_red={"3x3": 48, "5x5": 16}, c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16}, act_fn=self.act_fn, init_kernel=self.init_kernel, batch_norm=self.batch_norm),
        ]
        for block in inception_blocks:
            x = block(x, train=train) if isinstance(block, InceptionBlock) else block(x)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


class ResNetBlock(nn.Module):
    act_fn : Callable  # Activation function
    c_out : int   # Output feature size
    init_kernel: Callable # Kernel initializer
    batch_norm: bool = True # Whether to use batch normalization
    subsample : bool = False  # If True, we apply a stride inside F
    

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=self.init_kernel,
                    use_bias=False)(x)
        if self.batch_norm:
            z = nn.BatchNorm()(z, use_running_average=not train)
        else:
            z = nn.LayerNorm()(z)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=self.init_kernel,
                    use_bias=False)(z)
        if self.batch_norm:
            z = nn.BatchNorm()(z, use_running_average=not train)
        else:
            z = nn.LayerNorm()(z)

        if self.subsample:
            x = nn.Conv(self.c_out, kernel_size=(1, 1), strides=(2, 2), kernel_init=self.init_kernel)(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        if self.batch_norm:
            z = nn.BatchNorm()(x, use_running_average=not train)
        else:
            z = nn.LayerNorm()(x)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=self.init_kernel,
                    use_bias=False)(z)
        if self.batch_norm:
            z = nn.BatchNorm()(z, use_running_average=not train)
        else:
            z = nn.LayerNorm()(z)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=self.init_kernel,
                    use_bias=False)(z)

        if self.subsample:
            if self.batch_norm:
                z = nn.BatchNorm()(z, use_running_average=not train)
            else:
                z = nn.LayerNorm()(z)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=self.init_kernel,
                        use_bias=False)(x)

        x_out = z + x
        return x_out


class ResNet(nn.Module):
    num_classes : int
    act_fn : Callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)
    init_kernel: Callable = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x, train=True):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=self.init_kernel, use_bias=False)(x)
        if self.block_class == ResNetBlock:  # If pre-activation block, we do not apply non-linearities yet
            if self.batch_norm:
                x = nn.BatchNorm()(x, use_running_average=not train)
            else:
                x = nn.LayerNorm()(x)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(c_out=self.c_hidden[block_idx],
                                     act_fn=self.act_fn,
                                     subsample=subsample,
                                     init_kernel=self.init_kernel,
                                     batch_norm=self.batch_norm)(x, train=train)

        # Mapping to classification output
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


class DenseLayer(nn.Module):
    bn_size : int  # Bottleneck size (factor of growth rate) for the output of the 1x1 convolution
    growth_rate : int  # Number of output channels of the 3x3 convolution
    act_fn : Callable  # Activation function
    init_kernel: Callable  # Kernel initializer
    batch_norm: bool = True # Whether to use batch normalization

    @nn.compact
    def __call__(self, x, train=True):
        if self.batch_norm:
            z = nn.BatchNorm()(x, use_running_average=not train)
        else:
            z = nn.LayerNorm()(x)
        z = self.act_fn(z)
        z = nn.Conv(self.bn_size * self.growth_rate,
                    kernel_size=(1, 1),
                    kernel_init=self.init_kernel,
                    use_bias=False)(z)
        if self.batch_norm:
            z = nn.BatchNorm()(z, use_running_average=not train)
        else:
            z = nn.LayerNorm()(z)
        z = self.act_fn(z)
        z = nn.Conv(self.growth_rate,
                    kernel_size=(3, 3),
                    kernel_init=self.init_kernel,
                    use_bias=False)(z)
        x_out = jnp.concatenate([x, z], axis=-1)
        return x_out


class DenseBlock(nn.Module):
    num_layers : int  # Number of dense layers to apply in the block
    bn_size : int  # Bottleneck size to use in the dense layers
    growth_rate : int  # Growth rate to use in the dense layers
    act_fn : Callable  # Activation function to use in the dense layers
    init_kernel: Callable  # Kernel initializer
    batch_norm: bool = True # Whether to use batch normalization

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.num_layers):
            x = DenseLayer(bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           batch_norm=self.batch_norm,
                           init_kernel=self.init_kernel,
                           act_fn=self.act_fn)(x, train=train)
        return x


class TransitionLayer(nn.Module):
    c_out : int  # Output feature size
    act_fn : Callable  # Activation function
    init_kernel: Callable # Kernel initializer
    batch_norm: bool = True # Whether to use batch normalization

    @nn.compact
    def __call__(self, x, train=True):
        if self.batch_norm:
            x = nn.BatchNorm()(x, use_running_average=not train)
        else:
            x = nn.LayerNorm()(x)
        x = self.act_fn(x)
        x = nn.Conv(self.c_out,
                    kernel_size=(1, 1),
                    kernel_init=self.init_kernel,
                    use_bias=False)(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x


class DenseNet(nn.Module):
    num_classes : int
    act_fn : Callable = nn.relu
    num_layers : tuple = (6, 6, 6, 6)
    bn_size : int = 2
    growth_rate : int = 16
    init_kernel: Callable = nn.initializers.kaiming_normal()
    batch_norm: bool = True

    @nn.compact
    def __call__(self, x, train=True):
        c_hidden = self.growth_rate * self.bn_size  # The start number of hidden channels

        x = nn.Conv(c_hidden,
                    kernel_size=(3, 3),
                    kernel_init=self.init_kernel)(x)

        for block_idx, num_layers in enumerate(self.num_layers):
            x = DenseBlock(num_layers=num_layers,
                           bn_size=self.bn_size,
                           growth_rate=self.growth_rate,
                           batch_norm=self.batch_norm,
                           init_kernel=self.init_kernel,
                           act_fn=self.act_fn)(x, train=train)
            c_hidden += num_layers * self.growth_rate
            if block_idx < len(self.num_layers)-1:  # Don't apply transition layer on last block
                x = TransitionLayer(c_out=c_hidden//2,
                                    init_kernel=self.init_kernel,
                                    batch_norm=self.batch_norm,
                                    act_fn=self.act_fn)(x, train=train)
                c_hidden //= 2

        if self.batch_norm:
            x = nn.BatchNorm()(x, use_running_average=not train)
        else:
            x = nn.LayerNorm()(x)
        x = self.act_fn(x)
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x