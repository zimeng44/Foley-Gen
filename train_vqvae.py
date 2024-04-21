import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.cuda.amp as amp
import wandb
from audio2cembed import Audio2CEmbed
from audio2mel import Audio2Mel
from datasets import (clas_dict, get_dataset_filelist,
                      get_dataset_filelist_from_csv)
from scheduler import CycleScheduler
from torch import nn, optim
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from vqvae import VQVAE

LOG_IMAGES = True


def train(epoch, loader, model, optimizer, device, latent_loss_weight=0.25, classification_loss_weight=1e-2):
    model.train()

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    mse_sum = 0
    n = 0
    latent_sum = 0
    classification_sum = 0

    scaler = amp.GradScaler()

    for i, batch in enumerate(loader):
        model.zero_grad()
        if args.mel_only:
            embed, class_id, *_ = batch
            embed = embed.to(device, non_blocking=True)
            label = embed
        else:
            embed, label, class_id, *_ = batch
            embed = embed.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

        if args.unconditional:
            class_id = None
        else:
            class_id = class_id.to(device, non_blocking=True)

        out, latent_loss, classification_loss = model(embed, labels=class_id)
        recon_loss = criterion(out, label)

        with amp.autocast():
            loss = (
                    recon_loss
                    + latent_loss_weight * latent_loss.mean()
                    + classification_loss_weight * classification_loss.mean()
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mse_sum += recon_loss.item() * embed.shape[0]
        latent_sum += latent_loss.sum().item()
        classification_sum += classification_loss.sum().item()
        n += embed.shape[0]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(f"(Train) Epoch: {epoch + 1}")
        loader.set_postfix(
            mse=f"{recon_loss.item():.3e}",
            latent=f"{latent_loss.item():.3e}",
            classif=f"{classification_loss.item():.3e}",
            avg_mse=f"{mse_sum / n:.3e}",
            lr=f"{lr:.3e}",
        )

        # del embed, label, out, latent_loss, classification_loss, recon_loss, loss
        # torch.cuda.empty_cache()

    avg_latent_loss = latent_sum / n
    avg_classification_loss = classification_sum / n
    avg_mse = mse_sum / n

    torch.cuda.empty_cache()

    return avg_latent_loss, avg_mse, avg_classification_loss


@torch.inference_mode()
def validate(epoch, loader, model, device):
    model.eval()

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    mse_sum = 0
    n = 0
    latent_sum = 0
    classification_sum = 0

    for i, (embed, class_id, *_) in enumerate(loader):
        embed = embed.to(device)
        
        if not args.mel_only:
            class_id = class_id.to(device)
        else:
            class_id = None

        out, latent_loss, classification_loss = model(embed, labels=class_id)
        recon_loss = criterion(out, embed)

        mse_sum += recon_loss.item() * embed.shape[0]  # img.shape[0] = batch_size
        latent_sum += latent_loss.sum().item()
        classification_sum += classification_loss.sum().item()
        n += embed.shape[0]

        loader.set_description(f"(Validate) Epoch: {epoch + 1}")
        loader.set_postfix(
            mse=f"{recon_loss.item():.3e}",
            latent=f"{latent_loss.item():.3e}",
            classif=f"{classification_loss.item():.3e}",
            avg_mse=f"{mse_sum / n:.3e}",
        )

        del embed, out, latent_loss, classification_loss, recon_loss
        torch.cuda.empty_cache()

    avg_latent_loss = latent_sum / n
    avg_classification_loss = classification_sum / n
    avg_mse = mse_sum / n

    torch.cuda.empty_cache()

    return avg_latent_loss, avg_mse, avg_classification_loss


@torch.inference_mode()
def test(epoch, loader, model, device):
    model.eval()

    criterion = nn.MSELoss()

    mse_sum = 0
    mse_n = 0

    for i, (embed, *_) in enumerate(loader):
        embed = embed.to(device)

        out, latent_loss = model(embed)
        recon_loss = criterion(out, embed)
        latent_loss = latent_loss.mean()

        part_mse_sum = recon_loss.item() * embed.shape[0]  # img.shape[0] = batch_size
        part_mse_n = embed.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

            # validation
            if i % 100 == 0:
                pass

        del embed, out, latent_loss, recon_loss
        torch.cuda.empty_cache()

    latent_diff = latent_loss.item()
    if (epoch + 1) % 10 == 0:
        print(
            f"\nTest_Epoch: {epoch + 1}; "
            f"latent: {latent_diff:.3f}; Avg MSE: {mse_sum / mse_n:.5f} \n"
        )

    torch.cuda.empty_cache()

    return latent_diff, (mse_sum / mse_n)


def main(args):
    device = "cuda"

    # Logging into WanDB via API
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    # Creating wandb run
    run = wandb.init(
        name=os.environ.get("WANDB_VQVAE_RUN_NAME"),
        reinit=True,
        project=os.environ.get("WANDB_PROJECT"),
        config=vars(args)
    )

    if (args.valid_per_cat == None):
        valid_per_cat = 0
    else:
        valid_per_cat = args.valid_per_cat

    if (args.test_per_cat == None):
        test_per_cat = 0
    else:
        test_per_cat = args.test_per_cat

    if (args.load_split_data != None):
        train_file_list, valid_file_list, test_file_list = get_dataset_filelist_from_csv(args.load_split_data)
    else:
        train_file_list, valid_file_list, test_file_list = get_dataset_filelist(
            valid_per_cat, test_per_cat
        )

    if len(train_file_list) != 0:
        if not args.mel_only:
            train_set = Audio2CEmbed(train_file_list, 22050 * 4, 22050, is_train=True)
            collate_fn = train_set.collate_augment

        else:
            train_set = Audio2Mel(train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000)
            collate_fn = None

        train_loader = DataLoader(train_set,
                                batch_size=args.batch // args.n_gpu,
                                num_workers=8,
                                collate_fn=collate_fn,
                                pin_memory=True,
                                shuffle=True)
        print("training set size: " + str(len(train_set)))

    if len(valid_file_list) != 0:
        if not args.mel_only:
            valid_set = Audio2CEmbed(valid_file_list, 22050 * 4, 22050, is_train=False)
            collate_fn = valid_set.collate_augment
        else:
            valid_set = Audio2Mel(valid_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000)
            collate_fn = None

        valid_loader = DataLoader(valid_set,
                                  batch_size=args.batch // args.n_gpu,
                                  num_workers=8,
                                  collate_fn=collate_fn,
                                  pin_memory=True,
                                  shuffle=True)
        print("validation set size: " + str(len(valid_set)))

    if len(test_file_list) != 0:
        if not args.mel_only:
            test_set = Audio2CEmbed(test_file_list, 22050 * 4, 22050, is_train=False)
            collate_fn = test_set.collate_augment
        else:
            test_set = Audio2Mel(test_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000)
            collate_fn = None
        test_loader = DataLoader(test_set,
                                 batch_size=args.batch // args.n_gpu,
                                 num_workers=8,
                                 collate_fn=collate_fn,
                                 pin_memory=True,
                                 shuffle=True)
        print("test set size: " + str(len(test_set)))

    if args.model_config is None:
        model = VQVAE(n_classes=len(clas_dict))
    else:
        model = VQVAE(**args.model_config)

    if args.n_gpu > 1:
        model = DataParallel(model)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    model.enc_b = torch.compile(model.enc_b)
    model.quantize_conv_b = torch.compile(model.quantize_conv_b)
    model.dec = torch.compile(model.dec)

    if args.ckpt is not None:
        print(f"loading checkpoint from {args.ckpt}")
        # _, start_point = os.path.basename(args.ckpt).split("_")
        # start_point = int(start_point[0:-3])

        ckpt = torch.load(args.ckpt)
        start_point = ckpt["epoch"] + 1
        args = ckpt["args"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_point = 0

    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(train_loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )
    elif args.sched == "rlrop":
        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.8, patience=5 * len(train_loader), threshold=1e-04, cooldown=3 * len(train_loader),
            min_lr=1e-07
        )
    elif args.sched == "ca":
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)


    min_val_latent_diff, min_val_avg_loss, min_val_classif_loss = 1e12, 1e12, 1e12
    for i in range(start_point, args.epoch - start_point):
        train_latent_diff, train_average_loss, train_classif_loss = train(i, train_loader, model, optimizer, device,
                                                                          latent_loss_weight=args.latent_loss_weight,
                                                                          classification_loss_weight=args.classif_loss_weight)

        val_latent_diff, val_average_loss, val_classif_loss = validate(i, valid_loader, model, device)

        scheduler.step()

        curr_lr = optimizer.param_groups[0]['lr']

        torch.save(
            {
                "model": model.state_dict(),
                "args": args,
                "epoch": i,
                "optimizer": optimizer.state_dict(),
            },
            f"{args.ckpt_dir}/vqvae_epoch.pt",
        )
        wandb.save(f"{args.ckpt_dir}/vqvae_epoch.pt")

        if val_latent_diff < min_val_latent_diff:
            min_val_latent_diff = val_latent_diff
            print("Saving best valid_latent_diff model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": args,
                    "epoch": i,
                    "optimizer": optimizer.state_dict(),
                },
                f"{args.ckpt_dir}/vqvae_min_ldiff.pt",
            )
            wandb.save(f"{args.ckpt_dir}/vqvae_min_ldiff.pt")

        if val_average_loss < min_val_avg_loss:
            min_val_avg_loss = val_average_loss
            print("Saving best valid_average_loss model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": args,
                    "epoch": i,
                    "optimizer": optimizer.state_dict(),
                },
                f"{args.ckpt_dir}/vqvae_min_avgl.pt",
            )
            wandb.save(f"{args.ckpt_dir}/vqvae_min_avgl.pt")

        if val_classif_loss < min_val_classif_loss:
            min_val_classif_loss = val_classif_loss
            print("Saving best valid_classification_loss model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": args,
                    "epoch": i,
                    "optimizer": optimizer.state_dict(),
                },
                f"{args.ckpt_dir}/vqvae_min_classl.pt",
            )
            wandb.save(f"{args.ckpt_dir}/vqvae_min_classl.pt")

        cembedding_figures = {}
        if LOG_IMAGES and not args.mel_only:
            with torch.no_grad():
                model.eval()
                # Log actual vs reconstructed cembeddings for first 4 samples in train dataset
                outputs = [train_set[_] for _ in range(4)]
                cembed_augmented, cembed, class_ids, saliences, file_paths = train_set.collate_augment(outputs)
                cembed_augmented = cembed_augmented.to(device)
                cembed = cembed.to(device)
                class_ids = class_ids.to(device)

                decoded, _, _ = model(cembed_augmented)

                for j in range(4):
                    augment_spec = cembed_augmented[j].detach().cpu().numpy().squeeze(0)
                    actual_spec = cembed[0].detach().cpu().numpy().squeeze(0)
                    recon_spec = decoded[0].detach().cpu().numpy().squeeze(0)
                    fig, ax = plt.subplots(3, 1, figsize=(8, 9))
                    ax[0].imshow(actual_spec, aspect="auto")
                    ax[0].set_title("Actual Spectrogram")
                    ax[1].imshow(augment_spec, aspect="auto")
                    ax[1].set_title("Augmented Spectrogram")
                    ax[2].imshow(recon_spec, aspect="auto")
                    ax[2].set_title("Reconstructed Spectrogram")
                    # set title with epoch
                    fig.suptitle(f"Epoch {i}")
                    cembedding_figures[f"Training Output {j}"] = wandb.Image(fig)

                # Log actual vs reconstructed cembeddings for first 4 samples in validation dataset
                outputs = [valid_set[_] for _ in range(4)]
                cembed, class_ids, saliences, file_paths = valid_set.collate_augment(outputs)
                cembed = cembed.to(device)
                class_ids = class_ids.to(device)

                decoded, _, _ = model(cembed)
                for j in range(4):
                    actual_spec = cembed[j].detach().cpu().numpy().squeeze(0)
                    recon_spec = decoded[j].detach().cpu().numpy().squeeze(0)
                    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
                    ax[0].imshow(actual_spec, aspect="auto")
                    ax[0].set_title("Actual Spectrogram")
                    ax[1].imshow(recon_spec, aspect="auto")
                    ax[1].set_title("Reconstructed Spectrogram")
                    fig.suptitle(f"Epoch {i}")
                    cembedding_figures[f"Validation Output {j}"] = wandb.Image(fig)

        wandb.log(
            {"train_avg_loss": train_average_loss, 'train_latent_diff': train_latent_diff, "learning_Rate": curr_lr,
             "train_classif_loss": train_classif_loss,
             "val_avg_loss": val_average_loss, 'val_latent_diff': val_latent_diff,
             "val_classif_loss": val_classif_loss,
             **cembedding_figures
             })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )

    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--epoch", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--sched", type=str, default='ca')
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--latent_loss_weight", type=float, default=0.25)
    parser.add_argument("--classif_loss_weight", type=float, default=0.01)
    parser.add_argument("--mel_only", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint/vqvae/")
    parser.add_argument("--unconditional", action="store_true", help="Do not compute classification loss")

    # valid_per_cat: number of samples from each category to be used in the validation set
    parser.add_argument("--valid_per_cat", type=int, default=35)
    # test_per_cat: number of samples from each category to be used in the test set
    # the rest will be used in the training set
    parser.add_argument("--test_per_cat", type=int, default=35)
    # if no argument specified, a new "dataset_splits.csv" will be created. If specified, info will be
    # taken from the given file in the format {filename},{category} to create dataloaders
    parser.add_argument("--load_split_data", type=str)
    
    parser.add_argument("--config_json", type=str, default="")

    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.config_json:
        with open(args.config_json) as f:
            config = json.load(f)

        args.model_config = config
    else:
        args.model_config = None

    print(args)

    main(args)
