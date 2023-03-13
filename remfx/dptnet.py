import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.normalization import LayerNorm
from torch.autograd import Variable
import copy
import math


# adapted from https://github.com/ujscjj/DPTNet


class DPTNet_base(nn.Module):
    def __init__(
        self,
        enc_dim,
        feature_dim,
        hidden_dim,
        layer,
        segment_size=250,
        nspk=2,
        win_len=2,
    ):
        super().__init__()
        # parameters
        self.window = win_len
        self.stride = self.window // 2

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size

        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        self.dpt_encoder = DPTEncoder(
            n_filters=enc_dim,
            window_size=win_len,
        )
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)
        self.dpt_separation = DPTSeparation(
            self.enc_dim,
            self.feature_dim,
            self.hidden_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
        )

        self.mask_conv1x1 = nn.Conv1d(self.feature_dim, self.enc_dim, 1, bias=False)
        self.decoder = DPTDecoder(n_filters=enc_dim, window_size=win_len)

    def forward(self, batch):
        """
        mix: shape (batch, T)
        """
        mix, target = batch
        batch_size = mix.shape[0]
        mix = self.dpt_encoder(mix)  # (B, E, L)

        score_ = self.enc_LN(mix)  # B, E, L
        score_ = self.dpt_separation(score_)  # B, nspk, T, N
        score_ = (
            score_.view(batch_size * self.num_spk, -1, self.feature_dim)
            .transpose(1, 2)
            .contiguous()
        )  # B*nspk, N, T
        score = self.mask_conv1x1(score_)  # [B*nspk, N, L] -> [B*nspk, E, L]
        score = score.view(
            batch_size, self.num_spk, self.enc_dim, -1
        )  # [B*nspk, E, L] -> [B, nspk, E, L]
        est_mask = F.relu(score)

        est_source = self.decoder(
            mix, est_mask
        )  # [B, E, L] + [B, nspk, E, L]--> [B, nspk, T]

        return est_source


class DPTEncoder(nn.Module):
    def __init__(self, n_filters: int = 64, window_size: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(
            1, n_filters, kernel_size=window_size, stride=window_size // 2, bias=False
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x))
        return x


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, nhead, hidden_size, dim_feedforward, dropout, activation="relu"
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of improved part
        self.lstm = LSTM(d_model, hidden_size, 1, bidirectional=True)
        self.dropout = Dropout(dropout)
        self.linear = Linear(hidden_size * 2, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear(self.dropout(self.activation(self.lstm(src)[0])))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class SingleTransformer(nn.Module):
    """
    Container module for a single Transformer layer.
    args: input_size: int, dimension of the input feature.
    The input should have shape (batch, seq_len, input_size).
    """

    def __init__(self, input_size, hidden_size, dropout):
        super(SingleTransformer, self).__init__()
        self.transformer = TransformerEncoderLayer(
            d_model=input_size,
            nhead=4,
            hidden_size=hidden_size,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
        )

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        transformer_output = (
            self.transformer(output.permute(1, 0, 2).contiguous())
            .permute(1, 0, 2)
            .contiguous()
        )
        return transformer_output


# dual-path transformer
class DPT(nn.Module):
    """
    Deep dual-path transformer.
    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        num_layers: int, number of stacked Transformer layers. Default is 1.
        dropout: float, dropout ratio. Default is 0.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(DPT, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # dual-path transformer
        self.row_transformer = nn.ModuleList([])
        self.col_transformer = nn.ModuleList([])
        for i in range(num_layers):
            self.row_transformer.append(
                SingleTransformer(input_size, hidden_size, dropout)
            )
            self.col_transformer.append(
                SingleTransformer(input_size, hidden_size, dropout)
            )

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size, output_size, 1))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_transformer)):
            row_input = (
                output.permute(0, 3, 2, 1)
                .contiguous()
                .view(batch_size * dim2, dim1, -1)
            )  # B*dim2, dim1, N
            row_output = self.row_transformer[i](row_input)  # B*dim2, dim1, H
            row_output = (
                row_output.view(batch_size, dim2, dim1, -1)
                .permute(0, 3, 2, 1)
                .contiguous()
            )  # B, N, dim1, dim2
            output = row_output

            col_input = (
                output.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size * dim1, dim2, -1)
            )  # B*dim1, dim2, N
            col_output = self.col_transformer[i](col_input)  # B*dim1, dim2, H
            col_output = (
                col_output.view(batch_size, dim1, dim2, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # B, N, dim1, dim2
            output = col_output

        output = self.output(output)  # B, output_size, dim1, dim2

        return output


# base module for deep DPT
class DPT_base(nn.Module):
    def __init__(
        self, input_dim, feature_dim, hidden_dim, num_spk=2, layer=6, segment_size=250
    ):
        super(DPT_base, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.layer = layer
        self.segment_size = segment_size
        self.num_spk = num_spk

        self.eps = 1e-8

        # bottleneck
        self.BN = nn.Conv1d(self.input_dim, self.feature_dim, 1, bias=False)

        # DPT model
        self.DPT = DPT(
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim * self.num_spk,
            num_layers=layer,
        )

    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, dim, segment_stride)).type(
            input.type()
        )
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def split_feature(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = (
            input[:, :, :-segment_stride]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments2 = (
            input[:, :, segment_stride:]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )

        return segments.contiguous(), rest

    def merge_feature(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = (
            input.transpose(2, 3)
            .contiguous()
            .view(batch_size, dim, -1, segment_size * 2)
        )  # B, N, K, L

        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )
        input2 = (
            input[:, :, :, segment_size:]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, :-segment_stride]
        )

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]

        return output.contiguous()  # B, N, T

    def forward(self, input):
        pass


class DPTSeparation(DPT_base):
    def __init__(self, *args, **kwargs):
        super(DPTSeparation, self).__init__(*args, **kwargs)

        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, 1), nn.Sigmoid()
        )

    def forward(self, input):
        # input = input.to(device)
        # input: (B, E, T)
        batch_size, E, seq_length = input.shape

        enc_feature = self.BN(input)  # (B, E, L)-->(B, N, L)
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(
            enc_feature, self.segment_size
        )  # B, N, L, K: L is the segment_size
        # print('enc_segments.shape {}'.format(enc_segments.shape))
        # pass to DPT
        output = self.DPT(enc_segments).view(
            batch_size * self.num_spk, self.feature_dim, self.segment_size, -1
        )  # B*nspk, N, L, K

        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*nspk, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*nspk, K, T
        bf_filter = (
            bf_filter.transpose(1, 2)
            .contiguous()
            .view(batch_size, self.num_spk, -1, self.feature_dim)
        )  # B, nspk, T, N

        return bf_filter


class DPTDecoder(nn.Module):
    def __init__(self, n_filters: int = 64, window_size: int = 2):
        super().__init__()
        self.W = window_size
        self.basis_signals = nn.Linear(n_filters, window_size, bias=False)

    def forward(self, mixture, mask):
        """
        mixture: (batch, n_filters, L)
        mask: (batch, sources, n_filters, L)
        """
        source_w = torch.unsqueeze(mixture, 1) * mask  # [B, C, E, L]
        source_w = torch.transpose(source_w, 2, 3)  # [B, C, L, E]
        # S = DV
        est_source = self.basis_signals(source_w)  # [B, C, L, W]
        est_source = overlap_and_add(est_source, self.W // 2)  # B x C x T
        return est_source


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor.
        All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's
        inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(
        0, subframes_per_frame, subframe_step
    )
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result
