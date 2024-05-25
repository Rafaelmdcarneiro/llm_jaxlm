from ..types import PRNGKey, Array, DType, Shape
from dataclasses import field
import flax.linen as nn
from typing import Callable, Optional, Tuple


class Module(nn.Module):
    kernel_init: Callable = nn.initializers.variance_scaling
    kernel_init_args: tuple = (1.0, "fan_in", "truncated_normal")
    kernel_axes: Tuple[str, ...] = ()
    with_logical_partitioning: bool = True

    def setup(self):
        # wrap over init function in order to receive in_axis and out_axis
        def init_fn(
            key: PRNGKey, shape: Shape, dtype: DType, in_axis: int, out_axis: int
        ):
            fn = self.kernel_init(
                *self.kernel_init_args, in_axis=in_axis, out_axis=out_axis
            )
            if self.with_logical_partitioning:
                if not self.kernel_axes:
                    raise ValueError("with_logical_partitioning is True. Kernel axes must be specified.")
                fn = nn.with_logical_partitioning(fn, self.kernel_axes)
            return fn(key, shape, dtype)

        self.wrapped_kernel_init = init_fn


