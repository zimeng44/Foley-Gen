
import argparse
import json
import os
import pickle

import audio2cembed
import audio2mel
import lmdb
import torch
from datasets import (CodeRow, get_dataset_filelist,
                      get_dataset_filelist_from_csv)
from torch.utils.data import DataLoader
from tqdm import tqdm
from vqvae import VQVAE


@torch.inference_mode()
def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for batch in pbar:
            img, class_id, salience, filename = batch

            img = img.to(device)

            _, _, id_b = model.encode(img)
            id_b = id_b.detach().cpu().numpy()

            for c_id, sali, file, bottom in zip(class_id, salience, filename, id_b):
                row = CodeRow(
                    bottom=bottom, class_id=c_id, salience=sali, filename=file
                )
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vqvae_checkpoint', type=str, default='./checkpoint/vqvae/vqvae_min_avgl.pt'
    )
    parser.add_argument('--name', type=str, default='vqvae-code')
    parser.add_argument(
        "--dataset_splits", type=str, default="./dataset_splits.csv"
    )
    parser.add_argument(
        "--vqvae_config", type=str, default="./configs/vqvae_config_v0.json"
    )
    parser.add_argument("--mel_only", action="store_true", default=False, help="Do not use cembed")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    with open(args.vqvae_config) as f:
        config = json.load(f)

    device = 'cuda'

    train_file_list, val_file_list, test_file_list = get_dataset_filelist_from_csv(args.dataset_splits)

    if args.mel_only:
        train_set = audio2mel.Audio2Mel(
            train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
        )
        val_set = audio2mel.Audio2Mel(
            val_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
        )
        test_set = audio2mel.Audio2Mel(
            test_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
        )

        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=None, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=None, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=None, num_workers=2)
    else:
        train_set = audio2cembed.Audio2CEmbed(train_file_list, max_length=22050 * 4, is_train=False)
        val_set = audio2cembed.Audio2CEmbed(val_file_list, max_length=22050 * 4, is_train=False)
        test_set = audio2cembed.Audio2CEmbed(test_file_list, max_length=22050 * 4, is_train=False)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=None, num_workers=2, collate_fn=train_set.collate_augment)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=None, num_workers=2, collate_fn=val_set.collate_augment)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=None, num_workers=2, collate_fn=test_set.collate_augment)

    model = VQVAE(**config)
    checkpoint = torch.load(args.vqvae_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024

    os.makedirs(args.name, exist_ok=True)

    for loader, name in zip([train_loader, val_loader, test_loader], ['train', 'val', 'test']):
        dir_path = os.path.join(args.name, name)
        with lmdb.open(dir_path, map_size=map_size) as env:
            extract(env, loader, model, device)
