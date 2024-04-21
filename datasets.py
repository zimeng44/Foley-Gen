from typing import List

import os
import csv
import torch
import random
import numpy as np
import audio2mel
from vqvae import VQVAE
import lmdb
import pickle
from torch.utils.data import Dataset
from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['bottom', 'class_id', 'salience', 'filename'])
clas_dict: dict = {
    "DogBark": 0,
    "Footstep": 1,
    "Gunshot": 2,
    "Keyboard": 3,
    "MovingMotorVehicle": 4,
    "Rain": 5,
    "SneezeCough": 6,
}


def get_file_path(path):
    file_list = os.listdir(path)
    file_path = []
    for audio in file_list:
        # i.e., ['test_mel/7965-3-11-0.npy']
        file_path.append(os.path.join(path, audio))
    return file_path


def get_salience(file_name):
    annotation_file = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    _, filename = os.path.split(file_name)
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        result = result[1:]

        for row in result:
            if row[0] == filename:
                return row[4]


def get_class_id(file_name):
    annotation_file = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    _, filename = os.path.split(file_name)
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        result = result[1:]

        for row in result:
            if row[0] == filename:
                return row[6]


def get_dataset_filelist_urbansound8k(
        input_wavs_dir, input_annotation_file, test_fold_id: str, class_id=None
):
    training_files = []
    validation_files = []

    with open(input_annotation_file, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        for row in result[1:]:
            # slice_file_name, fsID, start, end, salience, fold, classID, class
            if class_id is None:
                if row[5] == test_fold_id:
                    validation_files.append(os.path.join(input_wavs_dir, row[0]))
                else:
                    training_files.append(os.path.join(input_wavs_dir, row[0]))
            else:
                if row[6] == class_id:

                    if row[5] == test_fold_id:
                        validation_files.append(os.path.join(input_wavs_dir, row[0]))
                    else:
                        training_files.append(os.path.join(input_wavs_dir, row[0]))

    return training_files, validation_files


def get_dataset_filelist_from_csv(csv_file):
    training_files: List[dict] = list()
    valid_files: List[dict] = list()
    test_files: List[dict] = list()
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            if row[1] == "train":
                training_files.append(
                    {
                        "class_id": clas_dict[row[0].split("/")[-2]],
                        "file_path": f"{row[0]}",
                    }
                )
            if row[1] == "validation":
                valid_files.append(
                    {
                        "class_id": clas_dict[row[0].split("/")[-2]],
                        "file_path": f"{row[0]}",
                    }
                )
            if row[1] == "test":
                test_files.append(
                    {
                        "class_id": clas_dict[row[0].split("/")[-2]],
                        "file_path": f"{row[0]}",
                    }
                )
    return training_files, valid_files, test_files


def get_dataset_filelist(num_valid, num_test, seed=0):
    training_files: List[dict] = list()
    valid_files: List[dict] = list()
    test_files: List[dict] = list()
    categories = [f.path for f in os.scandir("./DevSet") if f.is_dir()]

    with open("dataset_splits.csv", "w+") as f:
        f.write("filepath,split\n")
        f.close()

    random.seed(seed)

    for c in range(len(categories)):
        files = [f.path for f in os.scandir(categories[c])]
        num_files = len(files)
        num_train = len(files) - (num_valid + num_test)
        categ_dict = ["train"] * num_train + ["validation"] * num_valid + ["test"] * num_test
        random.shuffle(categ_dict)
        with open("dataset_splits.csv", "a") as csv_file:
            for f in range(num_files):
                if os.path.splitext(files[0])[-1] == ".wav":
                    if categ_dict[f] == "train":
                        training_files.append(
                            {
                                "class_id": clas_dict[files[f].split("/")[-2]],
                                "file_path": f"{files[f]}",
                            }
                        )
                    if categ_dict[f] == "validation":
                        valid_files.append(
                            {
                                "class_id": clas_dict[files[f].split("/")[-2]],
                                "file_path": f"{files[f]}",
                            }
                        )
                    if categ_dict[f] == "test":
                        test_files.append(
                            {
                                "class_id": clas_dict[files[f].split("/")[-2]],
                                "file_path": f"{files[f]}",
                            }
                        )

                    csv_file.write(str(files[f]) + "," + str(categ_dict[f]) + "\n")
    csv_file.close()
    return training_files, valid_files, test_files


"""
Generate Mels from Wav file
Args:
    source: wav_path 
    target: mel_path 
"""


def mel_extract(
        file_list,
        mel_path,
        max_length=22050 * 4,
        n_fft=1024,
        n_mels=80,
        hop_length=256,
        sample_rate=22050,
        fmin=0,
        fmax=8000,
):
    # file_list = get_file_path(wav_path)
    mel = audio2mel.Audio2Mel(
        file_list, max_length, n_fft, n_mels, hop_length, sample_rate, fmin, fmax
    )

    # print(mel[0][0].shape)

    for (item, _, _, filename) in mel:
        # print(item.shape)
        # item = F.interpolate(torch.tensor(item), scale_factor=2).numpy()
        np.save(os.path.join(mel_path, os.path.split(filename)[1]), item)
        print(filename, ' finished!')


def mel_extract_test():
    sample_size = 5
    test_file_list = get_dataset_filelist()[1]

    random.shuffle(test_file_list)

    test_file_list = test_file_list[0:sample_size]

    # extract mel of audio ground true
    mel_extract(test_file_list, 'test/test_mel')


def mel_generate_test():
    device = 'cuda:0'

    check_point = 'checkpoint/vqvae+/vqvae_560.pt'
    # check_point = 'checkpoint/small-vqvae2/vqvae_800.pt'

    model = VQVAE()
    model.load_state_dict(torch.load(check_point))
    model = model.to(device)
    model.eval()

    test_mel_list = get_file_path('/home/lxb/Desktop/sound-recognition/mel-test9')

    with torch.no_grad():
        for i, filname in enumerate(test_mel_list):
            # print("start " + i)
            mel = np.load(filname)
            mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
            # print(mel[0][0][0][-4:])
            out, _ = model(mel)
            out = out.squeeze(1).cpu().numpy()
            # print(out[0][0][-4:])
            np.save(
                os.path.join(
                    '/home/lxb/Desktop/sound-recognition/mel-generated9_ablation',
                    os.path.split(filname)[1],
                ),
                out,
            )


class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.bottom), row.class_id, row.salience, row.filename


if __name__ == '__main__':
    # sc = '/media/lxb/U-DISK/baseline_30w/baseline_30w/samples'
    # audio_list = get_file_path(sc)
    # des = 'sample-results/baseline_32_30w'
    # mel_extract(audio_list, des)
    mel_generate_test()

    # test_9_list = get_dataset_filelist()[1]
    # mel_extract(test_9_list, '/media/lxb/sound-recognition/mel-test9')

    # test_file_list = get_file_path('test/test_audio')
    #
    # mel_extract(test_file_list, 'test/test_mel')

    # mel_generate_test()

    # _, data = get_dataset_filelist(class_id='90')
    # print(len(data))

    # mel_extract_test()
    # dataset = LMDBDataset('code')
    #
    # loader = DataLoader(
    #     dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True
    # )
    # for i, (top, bottom, class_id, file_name) in enumerate(loader):
    #     print(top.shape, bottom.shape, len(class_id), len(file_name))

    # test to see latent space
    # with torch.no_grad():
    #     for i, filname in enumerate(test_mel_list):
    #         mel = np.load(filname)
    #         # x = torch.FloatTensor(x).unsqueeze(0).to(device)
    #         mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
    #         _, _, _, id_t, id_b = model.encode(mel)
    #         print(filname, id_b, id_t)
