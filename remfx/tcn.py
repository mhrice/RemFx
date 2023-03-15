# This code is based on the following repository written by Christian J. Steinmetz
# https://github.com/csteinmetz1/micro-tcn
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor

from remfx.utils import causal_crop, center_crop


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        crop_fn: Callable = causal_crop,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride

        self.crop_fn = crop_fn
        # Assumes stride of 1
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=True,
        )
        # residual connection
        self.res = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=1,
            groups=1,
            stride=stride,
            bias=False,
        )
        self.relu = nn.PReLU(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        x_in = x
        x = self.conv1(x)
        x = self.relu(x)

        # residual
        x_res = self.res(x_in)

        # causal crop
        x = x + self.crop_fn(x_res, x.shape[-1])

        return x


class TCN(nn.Module):
    def __init__(
        self,
        ninputs: int = 1,
        noutputs: int = 1,
        nblocks: int = 4,
        channel_growth: int = 0,
        channel_width: int = 32,
        kernel_size: int = 13,
        stack_size: int = 10,
        dilation_growth: int = 10,
        condition: bool = False,
        latent_dim: int = 2,
        norm_type: str = "identity",
        causal: bool = False,
        estimate_loudness: bool = False,
    ) -> None:
        super().__init__()
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.nblocks = nblocks
        self.channel_growth = channel_growth
        self.channel_width = channel_width
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.dilation_growth = dilation_growth
        self.condition = condition
        self.latent_dim = latent_dim
        self.norm_type = norm_type
        self.causal = causal
        self.estimate_loudness = estimate_loudness

        print(f"Causal: {self.causal}")
        if self.causal:
            self.crop_fn = causal_crop
        else:
            self.crop_fn = center_crop

        if estimate_loudness:
            self.loudness = torch.nn.Linear(latent_dim, 1)

        # audio model
        self.process_blocks = torch.nn.ModuleList()
        out_ch = -1
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            out_ch = in_ch * channel_growth if channel_growth > 1 else channel_width
            dilation = dilation_growth ** (n % stack_size)
            self.process_blocks.append(
                TCNBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation,
                    stride=1,
                    crop_fn=self.crop_fn,
                )
            )
        self.output = nn.Conv1d(out_ch, noutputs, kernel_size=1)

        # model configuration
        self.receptive_field = self.compute_receptive_field()
        self.block_size = 2048
        self.buffer = torch.zeros(2, self.receptive_field + self.block_size - 1)

    def forward(self, x: Tensor) -> Tensor:
        x_in = x
        for _, block in enumerate(self.process_blocks):
            x = block(x)
        # y_hat = torch.tanh(self.output(x))
        x_in = causal_crop(x_in, x.shape[-1])
        gain_ln = self.output(x)
        y_hat = torch.tanh(gain_ln * x_in)
        return y_hat

    def compute_receptive_field(self):
        """Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks):
            dilation = self.dilation_growth ** (n % self.stack_size)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf
