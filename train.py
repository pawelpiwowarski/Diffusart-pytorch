from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from train_helpers import read_yaml, tensors_to_pil, image_grid
import wandb
from models.Advanced_Conditional_Unet import Unet
import torch
from train_helpers import read_yaml, tensors_to_pil, image_grid
from pathlib import Path
from tqdm import tqdm
import os
import glob
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
from diffusers import DDPMScheduler
import lpips
import argparse
import torchinfo
import torch.nn.functional as F
import warnings


def main(
    training_batch_size,
    sampling_batch_size,
    image_size,
    epochs,
    diffusion_steps,
    name,
    dim_mults,
    use_convnext,
    config,
):
    accelerator = Accelerator()
    results_folder = Path("./results" + f"/{name}-size_{image_size}")
    results_folder.mkdir(exist_ok=True)
    device = accelerator.device

    save_model_every_epoch = config["save_model_every_epoch"]
    images_per_epoch_sampled = config["images_per_epoch_sampled"]
    channels = config["channels"]
    beta_1 = config["beta_1"]
    beta_T = config["beta_T"]
    ema_decay = config["ema_decay"]
    target_learning_rate = config["target_learning_rate"]
    checkpoints_to_keep = config["checkpoints_to_keep"]
    ddpm_sampling_steps = config["ddpm_sampling_steps"]
    clip_value = config["gradient_clipping"]
    name_of_the_dataset = config["name_of_the_dataset"]
    use_wandb = config["use_wandb"]
    nr_of_columns_in_sampling_image = config["nr_of_columns"]
    nr_of_rows_in_sampling_image = config["nr_of_rows"]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    def perform_transformations(examples):
        examples["full_colour_pixel_values"] = [
            transform(image.convert("RGB")) for image in examples["full_colour"]
        ]

        examples["sketch_pixel_values"] = [
            transform(image.convert("RGB")) for image in examples["sketch"]
        ]

        examples["sketch_and_scribbles_merged_pixel_values"] = [
            transform(image.convert("RGB"))
            for image in examples["sketch_and_scribbles_merged"]
        ]

        del examples["sketch"]
        del examples["full_colour"]
        del examples["sketch_and_scribbles_merged"]

        if "scribbles" in examples:
            del examples["scribbles"]

        return examples

    dataset = load_dataset(name_of_the_dataset, split="train").train_test_split(
        test_size=config["validation_split_size"]
    )

    # train dataset
    dataset_train = dataset["train"].with_transform(perform_transformations)
    # fixed validation dataset remains fixed throughout the training
    dataset_validation = dataset["test"].with_transform(perform_transformations)

    train_dataloader = DataLoader(
        dataset_train, batch_size=training_batch_size, shuffle=True
    )
    validation_dataloader = DataLoader(
        dataset_validation, batch_size=sampling_batch_size, shuffle=False
    )

    lpips_loss = lpips.LPIPS(net="alex").to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=diffusion_steps,
        beta_schedule="squaredcos_cap_v2",
        beta_start=beta_1,
        beta_end=beta_T,
    )
    noise_scheduler.set_timesteps(ddpm_sampling_steps, device=device)

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=dim_mults,
        use_convnext=use_convnext,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=target_learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=target_learning_rate,
        steps_per_epoch=len(train_dataloader),
        epochs=epochs,
    )

    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        validation_dataloader,
    ) = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, validation_dataloader
    )

    validation_batch = next(iter(validation_dataloader))

    if len(validation_batch["full_colour_pixel_values"]) < sampling_batch_size:
        raise Exception(
            f"Validation batch size is {len(validation_batch['full_colour_pixel_values'])} "
            f"but sampling batch size is {sampling_batch_size}, will cause issues."
            f"Change the sampling batch size or the validation split size."
        )

    if (
        nr_of_rows_in_sampling_image * nr_of_columns_in_sampling_image
        != sampling_batch_size
    ):
        raise Exception(
            f"nr_of_rows * nr_of_columns should be equal to sampling batch size. "
            f"Currently nr_of_rows * nr_of_columns is {nr_of_rows_in_sampling_image * nr_of_columns_in_sampling_image} "
            f"and sampling batch size is {sampling_batch_size}."
        )

    if use_wandb:
        wandb.init(
            project=f"{str(device)}_{name_of_the_dataset}",
            notes=str(
                torchinfo.summary(
                    model,
                    input_size=[
                        (training_batch_size, 3, image_size, image_size),
                        (training_batch_size,),
                        (training_batch_size, 3, image_size, image_size),
                        (training_batch_size, 3, image_size, image_size),
                    ],
                    device=device,
                )
            ),
            config={
                "image_size": image_size,
                "noise_scheduler": noise_scheduler.__class__.__name__,
                "diffusion_steps": diffusion_steps,
                "beta_1": beta_1,
                "beta_T": beta_T,
                "training_batch_size": training_batch_size,
                "sampling_batch_size": sampling_batch_size,
                "checkpoints_to_keep": checkpoints_to_keep,
                "save_model_every_epoch": save_model_every_epoch,
                "save_images_per_epoch_times": images_per_epoch_sampled,
                "save_images_every_step": len(train_dataloader)
                // images_per_epoch_sampled,
                "epochs": epochs,
                "main_optim": optimizer.__class__.__name__,
                "target_learning_rate": target_learning_rate,
                "ema_decay": ema_decay,
                "nr_train_examples": len(dataset_train),
                "nr_validation_examples": len(validation_batch),
                "dim_mults": dim_mults,
                "DDPMS_sampling_steps": ddpm_sampling_steps,
                "lr_scheduler": lr_scheduler.__class__.__name__,
                "name_of_the_dataset": name_of_the_dataset,
                "use_convnext": use_convnext,
                "gradient_clipping": clip_value,
            },
        )

    checkpoint_files = glob.glob(str(results_folder / "model-epoch_*.pt"))
    start_epoch = 0
    total_steps = 0
    if checkpoint_files:
        # Sort the list of matching files by modification time (newest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Select the newest file
        checkpoint_files = checkpoint_files[0]

        # Now, newest_model_file contains the path to the newest "model" file

        if use_wandb and not wandb.run.resumed:
            warnings.warn(
                """Found checkpoint file but the run is not in resume mode.
                  If you want to resume the run from this checkpoint please configure the --resume flag in wandb init."""
            )

        else:
            checkpoint = torch.load(checkpoint_files)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
            model.train()
            start_epoch = checkpoint["epoch"]
            total_steps = checkpoint["step"]
            print("Loaded model from checkpoint", checkpoint_files)

    else:
        print("No model files found in the folder. \n Training from scratch.")

    for epoch in range(start_epoch, start_epoch + epochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for step, train_batch in enumerate(tepoch):
                total_steps += 1
                tepoch.set_description(f"Epoch {epoch}")
                batch_size = train_batch["full_colour_pixel_values"].shape[0]
                sketch = train_batch["sketch_pixel_values"]
                full_colour = train_batch["full_colour_pixel_values"]
                scribbles = train_batch["sketch_and_scribbles_merged_pixel_values"]

                # Sample t uniformally for every example in the batch
                t = torch.randint(
                    0, diffusion_steps, (batch_size,), device=device
                ).long()

                # full colour is a tensor of shape (batch_size, 3, height, width)

                noise = torch.randn_like(full_colour, device=device)
                noisy_images = noise_scheduler.add_noise(full_colour, noise, t)

                noise_pred = model(
                    x=noisy_images,
                    time=t,
                    implicit_conditioning=scribbles,
                    explicit_conditioning=sketch,
                )

                loss = F.mse_loss(
                    noise_pred,
                    noise,
                    reduction="mean",
                )

                tepoch.set_postfix(loss=loss.item())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), clip_value, norm_type=2
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if use_wandb:
                    wandb.log(
                        {
                            "l2_loss": loss.item(),
                            "epoch": epoch,
                            "total_steps": total_steps,
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm
                            if accelerator.sync_gradients
                            else None,
                        }
                    )

                if (
                    total_steps % (len(train_dataloader) // images_per_epoch_sampled)
                    == 0
                ):
                    with torch.no_grad():
                        sketch = validation_batch["sketch_pixel_values"]
                        full_colour = validation_batch["full_colour_pixel_values"]
                        scribbles = validation_batch[
                            "sketch_and_scribbles_merged_pixel_values"
                        ]

                        b = full_colour.shape[0]

                        noise_for_plain = torch.randn_like(full_colour, device=device)

                        for _, t in tqdm(
                            enumerate(noise_scheduler.timesteps),
                            total=len(noise_scheduler.timesteps),
                        ):
                            noise_for_plain = noise_scheduler.scale_model_input(
                                noise_for_plain, t
                            ).to(device)

                            # extend t to batch size
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

                        plain_lpsis_normalised = torch.clamp(noise_for_plain, -1, 1)

                        plain_images = tensors_to_pil(noise_for_plain)

                        lpips_score_plain = lpips_loss(
                            plain_lpsis_normalised, full_colour
                        ).mean()

                        path_plain = results_folder / f"epoch_{epoch}_step_{step}.png"

                        # this is a grid of images for sampling batch size 16 (4x4)
                        # change the number rows and columns to change the grid if
                        # batch size is different
                        image_grid(
                            plain_images,
                            nr_of_rows_in_sampling_image,
                            nr_of_columns_in_sampling_image,
                        ).save(path_plain)

                    # save images to a folder inside results

                    # log images to wandb
                    if use_wandb:
                        wandb.log(
                            {
                                "DDPM_lpips_score_plain": lpips_score_plain.item(),
                                "epoch": epoch,
                                "total_steps": total_steps,
                                "plain_sampled_images": wandb.Image(str(path_plain)),
                            }
                        )

                # save model at the end of every epoch
                if (
                    len(tepoch) - 1 == step
                    and epoch % save_model_every_epoch == 0
                    and epoch != 0
                ):
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": total_steps,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict(),
                        },
                        str(results_folder / f"model-epoch_{epoch}.pt"),
                    )
                    if use_wandb:
                        wandb.save(str(results_folder / f"model-epoch_{epoch}.pt"))
                    tqdm.write(f"Saved model at epoch {epoch}")

                    checkpoint_files = glob.glob(
                        str(results_folder / "model-epoch_*.pt")
                    )

                    if len(checkpoint_files) > checkpoints_to_keep:
                        # Sort the list of matching files by modification time (newest first)
                        checkpoint_files.sort(
                            key=lambda x: os.path.getmtime(x), reverse=True
                        )
                        # Select the newest file
                        checkpoint_files = checkpoint_files[checkpoints_to_keep:]
                        # Now, newest_model_file contains the path to the newest "model" file
                        for checkpoint_file in checkpoint_files:
                            os.remove(checkpoint_file)
                            # remove from wandb (not supported via python)
                            # probably can be done via the API


if __name__ == "__main__":
    # read the configs from .config.yml file
    config = read_yaml("config.yaml")
    parser = argparse.ArgumentParser()
    # these are parameters that can be changed from the command line
    # others are read from the config.yml file
    parser.add_argument(
        "--training_batch_size", type=int, default=config["training_batch_size"]
    )
    parser.add_argument(
        "--sampling_batch_size", type=int, default=config["sampling_batch_size"]
    )
    parser.add_argument("--image_size", type=int, default=config["image_size"])
    parser.add_argument("--epochs", type=int, default=config["epochs"])
    parser.add_argument(
        "--diffusion_steps", type=int, default=config["diffusion_steps"]
    )
    parser.add_argument("--name", type=str, default=config["name_of_the_project"])
    parser.add_argument("--dim_mults", type=tuple, default=config["dim_mults"])
    parser.add_argument("--use_convnext", type=bool, default=config["use_convnext"])

    args = parser.parse_args()

    main(
        args.training_batch_size,
        args.sampling_batch_size,
        args.image_size,
        args.epochs,
        args.diffusion_steps,
        args.name,
        args.dim_mults,
        args.use_convnext,
        config,
    )
