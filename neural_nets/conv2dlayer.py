from typing import Tuple

import numpy as np


def conv2d(input, kernel, stride, padding):
    input_padded = np.pad(
        input,
        ((padding, padding), (padding, padding), (0, 0)),
        mode="constant",
    )

    input_height, input_width, _ = input_padded.shape
    kernel_height, kernel_width, _ = kernel.shape

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width))

    for y in range(0, output_height):
        for x in range(0, output_width):
            output[y, x] = np.sum(
                input_padded[
                    y * stride : y * stride + kernel_height,
                    x * stride : x * stride + kernel_width,
                ]
                * kernel
            )

    return output


class Conv2DLayer:
    def __init__(
        self,
        num_kernels: int,
        kernel_size: Tuple,
        stride: int,
        padding: int,
    ):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # init kernels w. rndm vals
        self.kernels = np.random.randn(
            num_kernels, *kernel_size, 1
        )  # assume 1 input channel

    def forward(self, input):
        """
        Perform a forward pass of the conv layer using the given input.

        Args:
        - input (numpy array): The input data (height x width x channels).

        Returns:
        - output (numpy array): The output of the layer.
        """
        _, input_height, input_width = input.shape

        output_height = (
            input_height - self.kernel_size[0] + 2 * self.padding
        ) // self.stride + 1
        output_width = (
            input_width - self.kernel_size[1] + 2 * self.padding
        ) // self.stride + 1

        output = np.zeros((self.num_kernels, output_height, output_width))

        for k in range(self.num_kernels):
            output[k, :, :] = conv2d(input, self.kernels[k], self.stride, self.padding)

        return output
