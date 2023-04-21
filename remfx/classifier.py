import torch
import torchaudio
import torch.nn as nn

# import hearbaseline

# import hearbaseline.vggish
# import hearbaseline.wav2vec2

import wav2clip_hear
import panns_hear


import torch.nn.functional as F
from remfx.utils import init_bn, init_layer


class PANNs(torch.nn.Module):
    def __init__(
        self, num_classes: int, sample_rate: float, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = panns_hear.load_model("hear2021-panns_hear.pth")
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=32000
        )
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(2048, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            x = self.resample(x)
            embed = panns_hear.get_scene_embeddings(x.view(x.shape[0], -1), self.model)
        return self.proj(embed)


class Wav2CLIP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: float,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = wav2clip_hear.load_model("")
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(512, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            x = self.resample(x)
            embed = wav2clip_hear.get_scene_embeddings(
                x.view(x.shape[0], -1), self.model
            )
        return self.proj(embed)


class VGGish(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: float,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        self.model = hearbaseline.vggish.load_model()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(128, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            x = self.resample(x)
            embed = hearbaseline.vggish.get_scene_embeddings(
                x.view(x.shape[0], -1), self.model
            )
        return self.proj(embed)


class wav2vec2(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: float,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        self.model = hearbaseline.wav2vec2.load_model()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(1024, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            x = self.resample(x)
            embed = hearbaseline.wav2vec2.get_scene_embeddings(
                x.view(x.shape[0], -1), self.model
            )
        return self.proj(embed)


# adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py


class Cnn14(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sample_rate: float,
        model_sample_rate: float,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        specaugment: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.model_sample_rate = model_sample_rate
        self.specaugment = specaugment

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            model_sample_rate,
            n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self.bn0 = nn.BatchNorm2d(n_mels)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True)

        self.init_weight()

        if sample_rate != model_sample_rate:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=model_sample_rate
            )

        if self.specaugment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(64, True)
            self.time_mask = torchaudio.transforms.TimeMasking(128, True)

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor, train: bool = False):
        """
        Input: (batch_size, data_length)"""

        if self.sample_rate != self.model_sample_rate:
            x = self.resample(x)

        x = self.melspec(x)

        if self.specaugment and train:
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(2, 1, sharex=True)
            # axs[0].imshow(x[0, :, :, :].detach().squeeze().cpu().numpy())
            x = self.freq_mask(x)
            x = self.time_mask(x)
            # axs[1].imshow(x[0, :, :, :].detach().squeeze().cpu().numpy())
            # plt.savefig("spec_augment.png", dpi=300)

        x = x.permute(0, 2, 1, 3)
        x = self.bn0(x)
        x = x.permute(0, 2, 1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=train)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=train)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=train)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=train)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=train)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=train)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=train)
        x = F.relu_(self.fc1(x))
        clipwise_output = self.fc_audioset(x)

        return clipwise_output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x
