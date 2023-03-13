# Adapted from https://github.com/AppleHolic/source_separation/tree/master/source_separation


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import single, concat_complex
from torch.nn.init import calculate_gain
from typing import Tuple
from scipy.signal import get_window
from librosa.util import pad_center


class ComplexConvBlock(nn.Module):
    """
    Convolution block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        layers: int = 4,
        bn_func=nn.BatchNorm1d,
        act_func=nn.LeakyReLU,
        skip_res: bool = False,
    ):
        super().__init__()
        # modules
        self.blocks = nn.ModuleList()
        self.skip_res = skip_res

        for idx in range(layers):
            in_ = in_channels if idx == 0 else out_channels
            self.blocks.append(
                nn.Sequential(
                    *[
                        bn_func(in_),
                        act_func(),
                        ComplexConv1d(in_, out_channels, kernel_size, padding=padding),
                    ]
                )
            )

    def forward(self, x: torch.tensor) -> torch.tensor:
        temp = x
        for idx, block in enumerate(self.blocks):
            x = block(x)

        if temp.size() != x.size() or self.skip_res:
            return x
        else:
            return x + temp


class SpectrogramUnet(nn.Module):
    def __init__(
        self,
        spec_dim: int,
        hidden_dim: int,
        filter_len: int,
        hop_len: int,
        layers: int = 3,
        block_layers: int = 3,
        kernel_size: int = 5,
        is_mask: bool = False,
        norm: str = "bn",
        act: str = "tanh",
    ):
        super().__init__()
        self.layers = layers
        self.is_mask = is_mask

        # stft modules
        self.stft = STFT(filter_len, hop_len)

        if norm == "bn":
            self.bn_func = nn.BatchNorm1d
        elif norm == "ins":
            self.bn_func = lambda x: nn.InstanceNorm1d(x, affine=True)
        else:
            raise NotImplementedError("{} is not implemented !".format(norm))

        if act == "tanh":
            self.act_func = nn.Tanh
            self.act_out = nn.Tanh
        elif act == "comp":
            self.act_func = ComplexActLayer
            self.act_out = lambda: ComplexActLayer(is_out=True)
        else:
            raise NotImplementedError("{} is not implemented !".format(act))

        # prev conv
        self.prev_conv = ComplexConv1d(spec_dim * 2, hidden_dim, 1)

        # down
        self.down = nn.ModuleList()
        self.down_pool = nn.MaxPool1d(3, stride=2, padding=1)
        for idx in range(self.layers):
            block = ComplexConvBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bn_func=self.bn_func,
                act_func=self.act_func,
                layers=block_layers,
            )
            self.down.append(block)

        # up
        self.up = nn.ModuleList()
        for idx in range(self.layers):
            in_c = hidden_dim if idx == 0 else hidden_dim * 2
            self.up.append(
                nn.Sequential(
                    ComplexConvBlock(
                        in_c,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bn_func=self.bn_func,
                        act_func=self.act_func,
                        layers=block_layers,
                    ),
                    self.bn_func(hidden_dim),
                    self.act_func(),
                    ComplexTransposedConv1d(
                        hidden_dim, hidden_dim, kernel_size=2, stride=2
                    ),
                )
            )

        # out_conv
        self.out_conv = nn.Sequential(
            ComplexConvBlock(
                hidden_dim * 2,
                spec_dim * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bn_func=self.bn_func,
                act_func=self.act_func,
            ),
            self.bn_func(spec_dim * 2),
            self.act_func(),
        )

        # refine conv
        self.refine_conv = nn.Sequential(
            ComplexConvBlock(
                spec_dim * 4,
                spec_dim * 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bn_func=self.bn_func,
                act_func=self.act_func,
            ),
            self.bn_func(spec_dim * 2),
            self.act_func(),
        )

    def log_stft(self, wav):
        # stft
        mag, phase = self.stft.transform(wav)
        return torch.log(mag + 1), phase

    def exp_istft(self, log_mag, phase):
        # exp
        mag = np.e**log_mag - 1
        # istft
        wav = self.stft.inverse(mag, phase)
        return wav

    def adjust_diff(self, x, target):
        size_diff = target.size()[-1] - x.size()[-1]
        assert size_diff >= 0
        if size_diff > 0:
            x = F.pad(
                x.unsqueeze(1), (size_diff // 2, size_diff // 2), "reflect"
            ).squeeze(1)
        return x

    def masking(self, mag, phase, origin_mag, origin_phase):
        abs_mag = torch.abs(mag)
        mag_mask = torch.tanh(abs_mag)
        phase_mask = mag / abs_mag

        # masking
        mag = mag_mask * origin_mag
        phase = phase_mask * (origin_phase + phase)
        return mag, phase

    def forward(self, wav):
        # stft
        origin_mag, origin_phase = self.log_stft(wav)
        origin_x = torch.cat([origin_mag, origin_phase], dim=1)

        # prev
        x = self.prev_conv(origin_x)

        # body
        # down
        down_cache = []
        for idx, block in enumerate(self.down):
            x = block(x)
            down_cache.append(x)
            x = self.down_pool(x)

        # up
        for idx, block in enumerate(self.up):
            x = block(x)
            res = F.interpolate(
                down_cache[self.layers - (idx + 1)],
                size=[x.size()[2]],
                mode="linear",
                align_corners=False,
            )
            x = concat_complex(x, res, dim=1)

        # match spec dimension
        x = self.out_conv(x)
        if origin_mag.size(2) != x.size(2):
            x = F.interpolate(
                x, size=[origin_mag.size(2)], mode="linear", align_corners=False
            )

        # refine
        x = self.refine_conv(concat_complex(x, origin_x))

        def to_wav(stft):
            mag, phase = stft.chunk(2, 1)
            if self.is_mask:
                mag, phase = self.masking(mag, phase, origin_mag, origin_phase)
            out = self.exp_istft(mag, phase)
            out = self.adjust_diff(out, wav)
            return out

        refine_wav = to_wav(x)

        return refine_wav


class RefineSpectrogramUnet(SpectrogramUnet):
    def __init__(
        self,
        spec_dim: int,
        hidden_dim: int,
        filter_len: int,
        hop_len: int,
        layers: int = 4,
        block_layers: int = 4,
        kernel_size: int = 3,
        is_mask: bool = True,
        norm: str = "ins",
        act: str = "comp",
        refine_layers: int = 1,
        add_spec_results: bool = False,
    ):
        super().__init__(
            spec_dim,
            hidden_dim,
            filter_len,
            hop_len,
            layers,
            block_layers,
            kernel_size,
            is_mask,
            norm,
            act,
        )
        self.add_spec_results = add_spec_results
        # refine conv
        self.refine_conv = nn.ModuleList(
            [
                nn.Sequential(
                    ComplexConvBlock(
                        spec_dim * 2,
                        spec_dim * 2,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bn_func=self.bn_func,
                        act_func=self.act_func,
                    ),
                    self.bn_func(spec_dim * 2),
                    self.act_func(),
                )
            ]
            * refine_layers
        )

    def forward(self, wav):
        # stft
        origin_mag, origin_phase = self.log_stft(wav)
        origin_x = torch.cat([origin_mag, origin_phase], dim=1)

        # prev
        x = self.prev_conv(origin_x)

        # body
        # down
        down_cache = []
        for idx, block in enumerate(self.down):
            x = block(x)
            down_cache.append(x)
            x = self.down_pool(x)

        # up
        for idx, block in enumerate(self.up):
            x = block(x)
            res = F.interpolate(
                down_cache[self.layers - (idx + 1)],
                size=[x.size()[2]],
                mode="linear",
                align_corners=False,
            )
            x = concat_complex(x, res, dim=1)

        # match spec dimension
        x = self.out_conv(x)
        if origin_mag.size(2) != x.size(2):
            x = F.interpolate(
                x, size=[origin_mag.size(2)], mode="linear", align_corners=False
            )

        # refine
        for idx, refine_module in enumerate(self.refine_conv):
            x = refine_module(x)
            mag, phase = x.chunk(2, 1)
            mag, phase = self.masking(mag, phase, origin_mag, origin_phase)
            if idx < len(self.refine_conv) - 1:
                x = torch.cat([mag, phase], dim=1)

        # clamp phase
        phase = phase.clamp(-np.pi, np.pi)

        out = self.exp_istft(mag, phase)
        out = self.adjust_diff(out, wav)

        if self.add_spec_results:
            out = (out, mag, phase)

        return out


class _ComplexConvNd(nn.Module):
    """
    Implement Complex Convolution
    A: real weight
    B: img weight
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.transposed = transposed

        self.A = self.make_weight(in_channels, out_channels, kernel_size)
        self.B = self.make_weight(in_channels, out_channels, kernel_size)

        self.reset_parameters()

    def make_weight(self, in_ch, out_ch, kernel_size):
        if self.transposed:
            tensor = nn.Parameter(torch.Tensor(in_ch, out_ch // 2, *kernel_size))
        else:
            tensor = nn.Parameter(torch.Tensor(out_ch, in_ch // 2, *kernel_size))
        return tensor

    def reset_parameters(self):
        # init real weight
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.A)

        # init A
        gain = calculate_gain("leaky_relu", 0)
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3.0) * std

        with torch.no_grad():
            # TODO: find more stable initial values
            self.A.uniform_(-bound * (1 / (np.pi**2)), bound * (1 / (np.pi**2)))
            #
            # B is initialized by pi
            # -pi and pi is too big, so it is powed by -1
            self.B.uniform_(-1 / np.pi, 1 / np.pi)


class ComplexConv1d(_ComplexConvNd):
    """
    Complex Convolution 1d
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        kernel_size = single(kernel_size)
        stride = single(stride)
        # edit padding
        padding = padding
        dilation = single(dilation)
        super(ComplexConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            single(0),
        )

    def forward(self, x):
        """
        Implemented complex convolution using combining 'grouped convolution' and
        'real / img weight'
        :param x: data (N, C, T) C is concatenated with C/2 real channels and C/2 idea channels
        :return: complex conved result
        """
        # adopt reflect padding
        if self.padding:
            x = F.pad(x, (self.padding, self.padding), "reflect")

        # forward real
        real_part = F.conv1d(
            x,
            self.A,
            None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=2,
        )

        # forward idea
        spl = self.in_channels // 2
        weight_B = torch.cat([self.B[:spl].data * (-1), self.B[spl:].data])
        idea_part = F.conv1d(
            x,
            weight_B,
            None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=2,
        )

        return real_part + idea_part


class ComplexTransposedConv1d(_ComplexConvNd):
    """
    Complex Transposed Convolution 1d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
    ):
        kernel_size = single(kernel_size)
        stride = single(stride)
        padding = padding
        dilation = single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            True,
            output_padding,
        )

    def forward(self, x, output_size=None):
        """
        Implemented complex transposed convolution using combining 'grouped convolution'
        and 'real / img weight'
        :param x: data (N, C, T) C is concatenated with C/2 real channels and C/2 idea channels
        :return: complex transposed convolution result
        """
        # forward real
        if self.padding:
            x = F.pad(x, (self.padding, self.padding), "reflect")

        real_part = F.conv_transpose1d(
            x,
            self.A,
            None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=2,
        )

        # forward idea
        spl = self.out_channels // 2
        weight_B = torch.cat([self.B[:spl] * (-1), self.B[spl:]])
        idea_part = F.conv_transpose1d(
            x,
            weight_B,
            None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=2,
        )

        if self.output_padding:
            real_part = F.pad(
                real_part, (self.output_padding, self.output_padding), "reflect"
            )
            idea_part = F.pad(
                idea_part, (self.output_padding, self.output_padding), "reflect"
            )

        return real_part + idea_part


class ComplexActLayer(nn.Module):
    """
    Activation differently 'real' part and 'img' part
    In implemented DCUnet on this repository, Real part is activated to log space.
    And Phase(img) part, it is distributed in [-pi, pi]...
    """

    def forward(self, x):
        real, img = x.chunk(2, 1)
        return torch.cat([F.leaky_relu_(real), torch.tanh(img) * np.pi], dim=1)


class STFT(nn.Module):
    """
    Re-construct stft for calculating backward operation
    refer on : https://github.com/pseeth/torch-stft/blob/master/torch_stft/stft.py
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 512,
        win_length: int = None,
        window: str = "hann",
    ):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.pad_amount = self.filter_length // 2

        # make fft window
        assert filter_length >= self.win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # calculate fourer_basis
        cut_off = int((self.filter_length / 2 + 1))
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cut_off, :]), np.imag(fourier_basis[:cut_off, :])]
        )

        # make forward & inverse basis
        self.register_buffer("square_window", fft_window**2)

        forward_basis = torch.FloatTensor(fourier_basis[:, np.newaxis, :]) * fft_window
        inverse_basis = (
            torch.FloatTensor(
                np.linalg.pinv(self.filter_length / self.hop_length * fourier_basis).T[
                    :, np.newaxis, :
                ]
            )
            * fft_window
        )
        # torch.pinverse has a bug, so at this time, it is separated into two parts..
        self.register_buffer("forward_basis", forward_basis)
        self.register_buffer("inverse_basis", inverse_basis)

    def transform(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # reflect padding
        wav = wav.unsqueeze(1).unsqueeze(1)
        wav = F.pad(
            wav, (self.pad_amount, self.pad_amount, 0, 0), mode="reflect"
        ).squeeze(1)

        # conv
        forward_trans = F.conv1d(
            wav, self.forward_basis, stride=self.hop_length, padding=0
        )
        real_part, imag_part = forward_trans.chunk(2, 1)

        return torch.sqrt(real_part**2 + imag_part**2), torch.atan2(
            imag_part.data, real_part.data
        )

    def inverse(
        self, magnitude: torch.Tensor, phase: torch.Tensor, eps: float = 1e-9
    ) -> torch.Tensor:
        comp = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        inverse_transform = F.conv_transpose1d(
            comp, self.inverse_basis, stride=self.hop_length, padding=0
        )

        # remove window effect
        n_frames = comp.size(-1)
        inverse_size = inverse_transform.size(-1)

        window_filter = torch.ones(1, 1, n_frames).type_as(inverse_transform)

        weight = self.square_window[: self.filter_length].unsqueeze(0).unsqueeze(0)
        window_filter = F.conv_transpose1d(
            window_filter, weight, stride=self.hop_length, padding=0
        )
        window_filter = window_filter.squeeze()[:inverse_size] + eps

        inverse_transform /= window_filter

        # scale by hop ratio
        inverse_transform *= self.filter_length / self.hop_length

        return inverse_transform[..., self.pad_amount : -self.pad_amount].squeeze(1)
