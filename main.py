from audio_diffusion_pytorch import AudioDiffusionModel
import torch
from tqdm import tqdm
import wandb

model = AudioDiffusionModel(in_channels=1)
wandb.init(project="RemFX", entity="mattricesound")

x = torch.randn(2, 1, 2**18)
for i in tqdm(range(100)):
    loss = model(x)
    loss.backward()
    if i % 10 == 0:
        print(loss)
        wandb.log({"loss": loss})


noise = torch.randn(2, 1, 2**18)
sampled = model.sample(noise=noise, num_steps=5)
