from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models.Advanced_Conditional_Unet import Unet
import torch
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
import os
import glob
from accelerate import Accelerator
from ema_pytorch import EMA
from train_helpers import read_yaml
from diffusers import DDPMScheduler
import lpips


accelerator = Accelerator()
device = accelerator.device
results_folder = Path("./results/test_samples")
results_folder.mkdir(exist_ok=True)

config = read_yaml("config.yaml")
channels = config["channels"]
image_size = config["image_size"]
timesteps = config["diffusion_steps"]
beta_1 = config["beta_1"]
beta_T = config["beta_T"]
sampling_batch_size = config["sampling_batch_size"]
ddpm_sampling_steps = config["ddpm_sampling_steps"]
use_convnext = config["use_convnext"]


transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)


def perform_transformations(examples):
    examples["sketch_pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["sketch"]
    ]
    examples["full_colour_pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["full_colour"]
    ]
    examples["sketch_and_scribbles_merged_pixel_values"] = [
        transform(image.convert("RGB"))
        for image in examples["sketch_and_scribbles_merged"]
    ]
    del examples["sketch"]
    del examples["full_colour"]
    del examples["scribbles"]
    del examples["sketch_and_scribbles_merged"]
    return examples


dataset_test = load_dataset("pawlo2013/anime_diffusion_full", split="test")


transformed_full_colour_dataset = dataset_test.with_transform(perform_transformations)
dataloader = DataLoader(
    transformed_full_colour_dataset,
    batch_size=sampling_batch_size,
    shuffle=False,
    num_workers=0,
)


model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4, 8),
    use_convnext=use_convnext,
).to(device)

lpips_loss = lpips.LPIPS(net="alex").to(device)

checkpoint_files = glob.glob(str(results_folder / "model-epoch_*.pt"))

if checkpoint_files:
    # Sort the list of matching files by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Select the newest file
    checkpoint_files = checkpoint_files[0]
    # Now, newest_model_file contains the path to the newest "model" file
    checkpoint = torch.load(checkpoint_files, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    model.eval()
    print("Loaded model from checkpoint", checkpoint_files)

else:
    raise Exception("No model files found in the folder.")

l_loss = 0
for idx, batch in enumerate(dataloader):
    sketch = batch["sketch_pixel_values"].to(device)
    scribbles = batch["sketch_and_scribbles_merged_pixel_values"].to(device)
    full_colour = batch["full_colour_pixel_values"].to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2"
    )
    noise_scheduler.set_timesteps(ddpm_sampling_steps, device=device)

    with torch.no_grad():
        # get only the first implicit conditioning

        b = sketch.shape[0]

        noise_for_plain = torch.randn_like(full_colour, device=device)

        for i, t in tqdm(
            enumerate(noise_scheduler.timesteps),
            total=len(noise_scheduler.timesteps),
        ):
            noise_for_plain = noise_scheduler.scale_model_input(noise_for_plain, t).to(
                device
            )

            time = t.expand(
                b,
            ).to(device)

            plain_noise_pred = model(
                x=noise_for_plain,
                time=time,
                implicit_conditioning=scribbles,
                explicit_conditioning=sketch,
            )

            noise_for_plain = noise_scheduler.step(
                plain_noise_pred,
                t.long(),
                noise_for_plain,
            ).prev_sample

        l_loss += lpips_loss(full_colour, torch.clamp(noise_for_plain, -1, 1)).mean()
        print("LPIPS Loss", l_loss / (idx + 1))
        images = torch.clamp((noise_for_plain / 2) + 0.5, 0, 1)
        implicit_conditioning = torch.clamp((scribbles / 2) + 0.5, 0, 1)
        full_colour = torch.clamp((full_colour / 2) + 0.5, 0, 1)
        sketch = torch.clamp((sketch / 2) + 0.5, 0, 1)

        images = torch.cat([sketch, implicit_conditioning, images, full_colour], dim=0)

        # save the images as columns one column for each implicit conditioning with the epoch number as the filename

        save_image(
            images,
            results_folder / f"sample-epoch_{idx}.png",
            nrow=sampling_batch_size,
        )

        print("Saved images")


print("LPIPS Loss", l_loss / len(dataloader))
