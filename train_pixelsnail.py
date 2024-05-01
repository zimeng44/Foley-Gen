from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import shutil
from typing import Optional
import math
import logging

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed
import torchinfo
import wandb
from datasets import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

MAX_CHECKPOINTS_TO_KEEP = 1
KEEP_EVERY = 10


def get_checkpoints_sorted(checkpoint_dir):
    paths = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    valid_paths = []
    epochs = []
    for path in paths:
        try:
            epoch = int(os.path.basename(path).split("_")[1].split(".")[0])
        except Exception:
            continue
        else:
            valid_paths.append(path)
            epochs.append(epoch)

    sorted_paths = [path for _, path in sorted(zip(epochs, valid_paths))]
    sorted_epochs = sorted(epochs)

    return sorted_paths, sorted_epochs


def remove_old_checkpoints(checkpoint_dir, keep_num=None, keep_every=None):
    paths, epochs = get_checkpoints_sorted(checkpoint_dir)

    # if len(epochs) > 1:
    #     if not np.all(np.diff(epochs) == 1):
    #         raise ValueError(
    #             f"Epochs are discontinuous, please clear old checkpoints"
    #             f"manually from {checkpoint_dir}. Received epochs: {epochs}."
    #         )

    if keep_num is not None:
        if len(paths) > max(keep_num, 1):
            for path in paths[:-keep_num]:
                os.remove(path)

    if keep_every is not None:
        if len(paths) > 1:
            for epoch, path in zip(epochs, paths):
                if epoch % keep_every != 0:
                    os.remove(path)


def train(
        epoch,
        loader,
        model,
        optimizer,
        scheduler,
        device,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        enable_amp: bool = False,
        max_gradient_norm=None
) -> tuple[float, float]:
    model.train()
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    valid_batches = 0

    for i, (bottom, class_id, salience, file_name) in enumerate(loader):
        optimizer.zero_grad()
        class_id = torch.FloatTensor(list(list(class_id))).long().unsqueeze(1)
        bottom = bottom.to(device, non_blocking=True)
        class_id = class_id.to(device, non_blocking=True)
        target = bottom

        with torch.cuda.amp.autocast(enabled=enable_amp):
            out, _ = model(bottom, label_condition=class_id)
            loss = criterion(out, target)

        accuracy = (out.max(1)[1] == target).float().sum() / target.numel()

        if not math.isnan(loss.item()):
            total_loss += loss.item()
            total_acc += accuracy.item()
            valid_batches += 1
            avg_loss = total_loss / valid_batches
            avg_acc = total_acc / valid_batches
        else:
            print(f"Loss is NaN, skipping batch {i}.")
            wandb.alert(title="Loss is NaN", text=f"Loss is NaN, epoch {epoch} batch {i}", level="ERROR")
            avg_loss = 0.0
            avg_acc = 0.0
            # stop wandb run
            wandb.finish()
            
            # stop training
            raise ValueError("Loss is NaN")

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(f"Training epoch: {epoch + 1:03d}")
        loader.set_postfix_str(
            (
                f"lr: {lr:.3E}; "
                f"avg_loss: {avg_loss:.5f}; avg_acc: {avg_acc:.5f}; "
                f"loss: {loss.item():.5f}; acc: {accuracy.item():.5f}" 
            )
        )

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_gradient_norm)
            optimizer.step()
        if scheduler is not None and isinstance(scheduler, CycleScheduler):
            scheduler.step(avg_loss)

        # del out, pred
        # torch.cuda.empty_cache()

    return avg_acc, avg_loss


@torch.no_grad()
def validate(loader, model, device):
    model.eval()
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    avg_loss = 0.0
    avg_acc = 0.0

    generated_samples = {}

    for i, (bottom, class_id, salience, filename) in enumerate(loader):
        class_id = torch.FloatTensor(list(list(class_id))).long().unsqueeze(1)
        bottom = bottom.to(device)
        class_id = class_id.to(device)

        target = bottom
        out, _ = model(bottom, label_condition=class_id)
        loss = criterion(out, target)

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        avg_acc += accuracy

        batch_loss = loss.item()
        loader.set_description(f"Validation: loss: {batch_loss:.5f}; acc: {accuracy:.5f}")

        avg_loss += float(batch_loss)

        # for j, (bottom_i, class_id_i, salience_i, filename_i) in enumerate(zip(bottom, class_id, salience, filename)):
        #     if class_id_i not in generated_samples:
        #         generated_samples[class_id_i] = {
        #             "filename": filename_i,
        #             "generated": out[j],
        #             "actual": target[j],
        #             "accuracy": (pred[j] == target[j]).float().sum() / target[j].numel(),
        #         }

        del bottom, class_id, target, out, loss, pred, correct, accuracy, batch_loss
        torch.cuda.empty_cache()
        gc.collect()

    avg_loss = float(avg_loss / len(loader))
    avg_acc = float(avg_acc / len(loader))

    return avg_acc, avg_loss


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == "__main__":
    os.makedirs("checkpoint/pixelsnail-final", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--hier", type=str, default="bottom")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--channel", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--n_block", type=int, default=4)
    parser.add_argument("--n_res_block", type=int, default=4)
    parser.add_argument("--n_res_channel", type=int, default=256)
    parser.add_argument("--n_out_res_block", type=int, default=0)
    parser.add_argument("--n_cond_res_block", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--sched", type=str, default=None)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--path", type=str, default="vqvae-code/")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument("--model_config_json", type=str, default=None)
    parser.add_argument("--n_gpu", type=int, default=None, help="If None, use all GPUs")
    parser.add_argument("--wandb_disabled", action="store_true", help="Disable wandb logging")
    parser.add_argument("--reset_lr", action="store_true", help="Reset learning rate when resuming from checkpoint")
    parser.add_argument("--max_gradient_norm", type=float, default=None, help="Max gradient norm for gradient clipping")

    args = parser.parse_args()
    reset_lr = args.reset_lr

    if args.wandb_disabled:
        os.environ["WANDB_DISABLED"] = "true"

    model_config_dict = None
    if args.model_config_json is not None:
        print(f"Loading model config from {args.model_config_json}.")
        with open(args.model_config_json) as f:
            model_config_dict = json.load(f)
            args.model_config_dict = model_config_dict

    device_count = torch.cuda.device_count()
    if args.n_gpu is None:
        args.n_gpu = device_count
    else:
        args.n_gpu = min(args.n_gpu, device_count)
    n_gpu = args.n_gpu

    print(args)

    device = "cuda"

    # Logging into WanDB via API
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    if args.wandb_resume_id is None:
        # Creating wandb run
        run = wandb.init(
            name=os.environ.get("WANDB_PIXELSNAIL_RUN_NAME"),
            reinit=True,
            project=os.environ.get("WANDB_PROJECT"),
            config=vars(args),
            entity=os.environ.get("WANDB_ENTITY"),
        )
    else:
        run = wandb.init(
            name=os.environ.get("WANDB_PIXELSNAIL_RUN_NAME"),
            reinit=True,
            project=os.environ.get("WANDB_PROJECT"),
            config=vars(args),
            entity=os.environ.get("WANDB_ENTITY"),
            resume="must",
            id=args.wandb_resume_id,
        )

    train_path = os.path.join(args.path, "train")
    val_path = os.path.join(args.path, "val")
    test_path = os.path.join(args.path, "test")

    train_dataset = LMDBDataset(train_path)
    val_dataset = LMDBDataset(val_path)
    test_dataset = LMDBDataset(test_path)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=8*device_count, drop_last=True, prefetch_factor=6, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=2*device_count, drop_last=False, prefetch_factor=2, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False, num_workers=2, drop_last=False
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    sched = args.sched
    max_gradient_norm = args.max_gradient_norm
    
    ckpt = {}
    start_point = 0

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        train_epoch = args.epoch
    
        sched = args.sched
        lr = args.lr

        args = ckpt["args"]
        args.epoch = train_epoch

        # FIXME: Overwrite args -- dirty
        if lr is not None:
            args.lr = lr

        if sched is not None:
            args.sched = sched
        
        if args.lr is None:
            args.lr = 1e-5
        
        if args.sched is None:
            args.sched = "cycle"

        start_point = ckpt["epoch"] + 1
        print(f"Resuming from args: {args}")
    else:
        if args.lr is None:
            args.lr = 1e-5

    if model_config_dict is None:
        model = PixelSNAIL(
            [288, 75],
            512,
            args.channel,
            kernel_size=args.kernel_size,
            n_block=args.n_block,
            n_res_block=args.n_res_block,
            res_channel=args.n_res_channel,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
            attention_downsample=2,
            use_skip_connect=True
        )
    else:
        model = PixelSNAIL(**model_config_dict)

    if "model" in ckpt:
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise e
            # state_dict = remap_checkpoint_keys(ckpt["model"])
            model.load_state_dict(state_dict)
        torch.cuda.empty_cache()

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.cuda.empty_cache()

    if reset_lr:
        optimizer.param_groups[0]["lr"] = args.lr

    model = nn.DataParallel(model)
    model = model.to(device)

    for batch in train_loader:
        bottom, class_id, salience, filename = batch
        with torch.cuda.amp.autocast():
            torchinfo.summary(model, input_data=[bottom, class_id])
        del bottom, class_id, salience, filename
        break

    scheduler = None
    if args.sched == "cycle":
        if "scheduler" in ckpt:
            scheduler = CycleScheduler.from_pickle(ckpt["scheduler"], optimizer=optimizer)
        else:
            scheduler = CycleScheduler(
                optimizer, args.lr, n_iter=len(train_loader) * args.epoch, momentum=None, divider=10
            )
            if hasattr(scheduler, "optimizer"):
                del scheduler.optimizer
                torch.cuda.empty_cache()
                scheduler.optimizer = optimizer

        print("Using Cycle scheduler")

    elif args.sched == "rlrop":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=0.8,
            patience=5 * len(train_loader),
            threshold=1e-03,
            cooldown=3 * len(train_loader),
            min_lr=1e-07,
            mode='min'
        )

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print("Using ReduceLROnPlateau scheduler")

    elif args.sched == "ca":
        print("Using CosineAnnealing scheduler")
        cosine_annealing_args = dict(T_max=10, eta_min=1e-5)
        print(f"Creating CosineAnnealingLR with args: {cosine_annealing_args}")
        scheduler = CosineAnnealingLR(optimizer, **cosine_annealing_args)

        if "scheduler" in ckpt:
            scheduler_state_dict = ckpt["scheduler"]
            scheduler.load_state_dict(scheduler_state_dict)
            print(f"Loading CosineAnnealingLR state dict: {scheduler_state_dict}")
        optimizer_lr = optimizer.param_groups[0]["lr"]
        if reset_lr:
            optimizer.param_groups[0]["lr"] = lr
            print(f"Resetting optimizer LR (--reset_lr) from {optimizer_lr} to {lr}")
        else:
            optimizer.param_groups[0]["lr"] = scheduler._last_lr[0]
            print(f"Setting optimizer LR {optimizer_lr} to last cycle scheduler LR {scheduler._last_lr[0]}.")
    else:
        print(f"Not using any scheduler. Unknown scheduler '{args.sched}'")
        scheduler = None

    torch.cuda.empty_cache()

    t_loss_best = np.inf
    v_loss_best = np.inf

    for group in optimizer.param_groups:
        group["weight_decay"] = 1e-3
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Start point: {start_point}")
    print(f"{args.epoch - start_point=}")
    for i in range(start_point, args.epoch - start_point):
        paths, epochs = get_checkpoints_sorted("checkpoint/pixelsnail-final")
        num_checkpoints = len(paths)
        # remove_old_checkpoints("checkpoint/pixelsnail-final", keep_every=KEEP_EVERY)

        t_acc, t_loss = train(
            i,
            train_loader,
            model,
            optimizer,
            scheduler,
            device,
            scaler=scaler,
            enable_amp=args.amp,
            max_gradient_norm=max_gradient_norm,
        )
        torch.cuda.empty_cache()

        curr_lr = optimizer.param_groups[0]["lr"]

        v_acc, v_loss = validate(val_loader, model, device)
        torch.cuda.empty_cache()


        if args.sched == 'rlrop':
            scheduler.step(v_loss)
        elif args.sched == 'ca':
            scheduler.step()

        wandb.log({"train_loss": t_loss, "train_Acc": t_acc, "learning_Rate": curr_lr, "val_loss": v_loss, "val_Acc": v_acc})

        # epoch_checkpoint_save_path = f"checkpoint/pixelsnail-final/{args.hier}_{str(i + 1).zfill(3)}.pt"
        latest_checkpoint_save_path = "checkpoint/pixelsnail-final/bottom_latest.pt"
        # best_checkpoint_save_path = "checkpoint/pixelsnail-final/bottom_best.pt"

        checkpoint_dict = {
            "model": model.state_dict(),
            "args": args,
            "epoch": i,
            "optimizer": optimizer.state_dict(),
            "t_acc": t_acc,
            "t_loss": t_loss,
            "v_acc": v_acc,
            "v_loss": v_loss,
            "curr_lr": curr_lr,
        }

        if scheduler is not None:
            if isinstance(scheduler, CycleScheduler):
                checkpoint_dict["scheduler"] = scheduler.to_pickle()
            elif hasattr(scheduler, "state_dict"):
                checkpoint_dict["scheduler"] = scheduler.state_dict()

        try:
            torch.save(checkpoint_dict, latest_checkpoint_save_path)
            wandb.save(latest_checkpoint_save_path, policy="live")
        except OSError as e:
            if "No space left on device" in str(e):
                print("No space left on disk. Removing one old checkpoint.")
                remove_old_checkpoints("checkpoint/pixelsnail-final", keep_num=max(1, num_checkpoints-2))
                torch.save(checkpoint_dict, latest_checkpoint_save_path)
                wandb.save(latest_checkpoint_save_path, policy="live")
            else:
                raise e

        # if t_loss < t_loss_best:
        #     t_loss_best = t_loss
        #     shutil.copy(epoch_checkpoint_save_path, best_checkpoint_save_path)
        #     wandb.save(best_checkpoint_save_path)

        # if v_loss < v_loss_best:
        #     v_loss_best = v_loss
        #     shutil.copy(epoch_checkpoint_save_path, best_checkpoint_save_path)
        #     wandb.save(best_checkpoint_save_path, policy="end")

        # shutil.copy(epoch_checkpoint_save_path, latest_checkpoint_save_path)
        # if i % KEEP_EVERY == 0:
        #     wandb.save(latest_checkpoint_save_path, policy="now")
