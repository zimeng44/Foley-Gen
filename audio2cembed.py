import os

import datasets
import librosa
import numpy as np
import torch
import torchaudio.transforms as T
from matplotlib import pyplot as plt
from scipy.io.wavfile import read as loadwav
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MAX_WAV_VALUE = 32768.0

""" Mel-Spectrogram extraction code from HiFi-GAN meldataset.py"""


def get_normalized_sample_name(file_path: str) -> str:
    file_name = os.path.basename(file_path)
    sample_name = os.path.splitext(file_name)[0]
    class_name = os.path.basename(os.path.dirname(file_path))
    return f"{class_name}_{sample_name}"


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram_hifi(
    audio, n_fft, n_mels, sample_rate, hop_length, fmin, fmax, center=False
):
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    if torch.min(audio) < -1.0:
        print("min value is ", torch.min(audio))
    if torch.max(audio) > 1.0:
        print("max value is ", torch.max(audio))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel_fb = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(audio.device)] = (
            torch.from_numpy(mel_fb).float().to(audio.device)
        )
        hann_window[str(audio.device)] = torch.hann_window(n_fft).to(audio.device)

    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    audio = audio.squeeze(1)

    spec = torch.stft(
        audio,
        n_fft,
        hop_length=hop_length,
        window=hann_window[str(audio.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    mel = torch.matmul(mel_basis[str(fmax) + "_" + str(audio.device)], spec)
    mel = spectral_normalize_torch(mel).numpy()
    return mel


""" Mel-Spectrogram extraction code from HiFi-GAN meldataset.py"""


class Audio2CEmbed(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_files,
        max_length,
        sample_rate=22050,
        n_fft=1024,
        n_mels=129,
        hop_length=320,
        fmin=0,
        fmax=10000,
        is_train=True,
        device="cuda",
        mert_save_dir="mert_embeddings",
        mel_save_dir="mel_spectrograms",
        low_memory=True,
        mert_mean=1.9993998,
        mert_std=5.2803254,
        mel_mean=None,  # -6.358811
        mel_std=None,  # 2.873877
    ):
        self.audio_files = audio_files
        self.max_length = int(max_length * 1.003)
        self.sample_rate = sample_rate
        self.salience = 1

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

        self.is_train = is_train

        self.device = device

        self.mert_save_dir = mert_save_dir
        self.mel_save_dir = mel_save_dir

        self.low_memory = low_memory
        
        self.mert_mean = mert_mean
        self.mert_std = mert_std

        if self.is_train:
            self.augmentations_melspec = torch.nn.Sequential(
                T.TimeMasking(time_mask_param=25, p=0.6),
                T.FrequencyMasking(freq_mask_param=25),
            )
            self.augmentations_mert = torch.nn.Sequential(
                T.TimeMasking(time_mask_param=25, p=0.6),
                T.FrequencyMasking(freq_mask_param=250),
            )

        self.mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        ).to(device)

        self.resample_rate = self.mert_processor.sampling_rate
        if self.resample_rate != self.sample_rate:
            print(f"setting rate from {self.sample_rate} to {self.resample_rate}")
            resampler = T.Resample(self.sample_rate, self.resample_rate)
        else:
            resampler = None

        os.makedirs(self.mert_save_dir, exist_ok=True)

        self.mert_embeds, self.class_ids = [], []
        self.mert_embed_paths = []
        for audio_file in tqdm(self.audio_files, desc="Obtaining MERT embeddings"):
            filename = audio_file["file_path"]
            class_id = audio_file["class_id"]
            sample_name = get_normalized_sample_name(filename)
            mert_save_path = os.path.join(mert_save_dir, sample_name + ".npy")

            if os.path.exists(mert_save_path):
                if not self.low_memory:
                    mert_numpy = np.load(mert_save_path)
            else:
                sample_rate, audio = loadwav(filename)
                if sample_rate != self.sample_rate:
                    raise ValueError(
                        "{} sr doesn't match {} sr ".format(
                            sample_rate, self.sample_rate
                        )
                    )

                audio = audio / MAX_WAV_VALUE
                if len(audio) > self.max_length:
                    audio = audio[0 : self.max_length]
                elif len(audio) < self.max_length:
                    # pad audio to max length, 4s for Urbansound8k dataset
                    audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")

                audio = torch.from_numpy(audio).float()
                if resampler is not None:
                    audio = resampler(audio)
                if torch.min(audio) < -1.0:
                    audio /= -torch.min(audio)
                if torch.max(audio) > 1.0:
                    audio /= torch.max(audio)

                mert_embed = self.apply_mert(audio)
                mert_numpy = mert_embed.cpu().numpy().astype(np.float32)

                np.save(mert_save_path, mert_numpy)

            if not self.low_memory:
                self.mert_embeds.append(mert_numpy)

            self.class_ids.append(class_id)
            self.mert_embed_paths.append(mert_save_path)

        if not self.low_memory:
            self.mert_embeds = torch.tensor(self.mert_embeds)

        self.class_ids = torch.tensor(self.class_ids)

        os.makedirs(self.mel_save_dir, exist_ok=True)

        self.melspecs = []
        self.melspec_paths = []
        for audio_file in tqdm(self.audio_files, desc="Obtaining mel spectrogram"):
            filename = audio_file["file_path"]
            sample_name = get_normalized_sample_name(filename)
            mel_save_path = os.path.join(mel_save_dir, sample_name + ".npy")

            if os.path.exists(mel_save_path):
                if not self.low_memory:
                    melspec = np.load(mel_save_path)
            else:
                sample_rate, audio = loadwav(filename)
                audio = audio / MAX_WAV_VALUE
                if len(audio) > self.max_length:
                    audio = audio[0 : self.max_length]
                elif len(audio) < self.max_length:
                    # pad audio to max length, 4s for Urbansound8k dataset
                    audio = np.pad(audio, (0, self.max_length - len(audio)), "constant")

                if resampler is not None:
                    audio = resampler(torch.from_numpy(audio).float())
                if torch.min(audio) < -1.0:
                    audio /= -torch.min(audio)
                if torch.max(audio) > 1.0:
                    audio /= torch.max(audio)

                melspec = mel_spectrogram_hifi(
                    audio,
                    n_fft=self.n_fft,
                    n_mels=self.n_mels,
                    hop_length=self.hop_length,
                    sample_rate=self.resample_rate,
                    fmin=self.fmin,
                    fmax=self.fmax,
                )

                melspec = melspec.astype(np.float32).squeeze(0)

                np.save(mel_save_path, melspec)

            if not self.low_memory:
                self.melspecs.append(melspec)

            self.melspec_paths.append(mel_save_path)

        if not self.low_memory:
            self.melspecs = torch.tensor(self.melspecs)

        # Freeing RAM
        del self.mert_processor
        del self.mert_model
        torch.cuda.empty_cache()

    def apply_mert(self, audio):
        processed_model_input = self.mert_processor(
            audio, sampling_rate=self.resample_rate, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            model_output = self.mert_model(
                **processed_model_input, output_hidden_states=True
            )
            all_layer_hidden_states = torch.stack(model_output.hidden_states).squeeze()
            spec = all_layer_hidden_states.mean(0).T

            # Removing index=(951, :) feature since it looks constantly erroneous for all samples
            spec = torch.cat((spec[:951, :], spec[952:, :]), axis=0)
        return spec

    def __getitem__(self, index):
        if not self.low_memory:
            mert_embed = self.mert_embeds[index, :, :]
            melspec = self.melspecs[index, :, :, :].squeeze()
        else:
            mert_embed = torch.from_numpy(np.load(self.mert_embed_paths[index]))
            melspec = torch.from_numpy(np.load(self.melspec_paths[index]))

        if self.mert_mean is not None:
            mert_embed -= self.mert_mean

        if self.mert_std is not None:
            mert_embed /= self.mert_std

        return (
            mert_embed,
            melspec,
            self.class_ids[index],
            self.salience,
            self.audio_files[index]["file_path"],
        )

    def collate_augment(self, batch):
        mert_embeds, melspecs, class_ids, saliences, file_paths = [], [], [], [], []
        for mert_embed, melspec, class_id, salience, file_path in batch:
            mert_embeds.append(mert_embed)
            melspecs.append(melspec)
            class_ids.append(class_id)
            saliences.append(salience)
            file_paths.append(file_path)
        mert_embeds = torch.stack(mert_embeds)
        melspecs = torch.stack(melspecs)
        class_ids = torch.tensor(class_ids)

        cembed = torch.cat((mert_embeds, melspecs), dim=1)
        cembed = torch.unsqueeze(cembed, dim=1)

        if self.is_train:
            mert_embeds_augmented = self.augmentations_mert(mert_embeds)
            melspecs_augmented = self.augmentations_melspec(melspecs)
            cembed_augmented = torch.cat(
                (mert_embeds_augmented, melspecs_augmented), dim=1
            )
            cembed_augmented = torch.unsqueeze(cembed_augmented, dim=1)
            return cembed_augmented, cembed, class_ids, saliences, file_paths

        return cembed, class_ids, saliences, file_paths

    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    (
        train_file_list,
        valid_file_list,
        test_file_list,
    ) = datasets.get_dataset_filelist_from_csv("dataset_splits.csv")

    print(train_file_list[100])

    train_set = Audio2CEmbed(train_file_list[0:5], 22050 * 4, 22050, is_train=True)
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        collate_fn=train_set.collate_augment,
        shuffle=True,
    )
    for i, (embed_input, embed_label, class_id, salience, filename) in enumerate(
        train_loader
    ):
        print(embed_input.shape)
        plt.imshow(embed_input[0][0], aspect="auto")
        plt.show()
        print(embed_label.shape)
        plt.imshow(embed_label[0][0], aspect="auto")
        plt.show()
        break

    valid_set = Audio2CEmbed(valid_file_list[0:5], 22050 * 4, 22050, is_train=False)
    valid_loader = DataLoader(
        valid_set,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        collate_fn=valid_set.collate_augment,
        shuffle=True,
    )
    for i, (embed, class_id, salience, filename) in enumerate(valid_loader):
        print(embed.shape)
        plt.imshow(embed[0][0], aspect="auto")
        plt.show()
        break

    test_set = Audio2CEmbed(test_file_list[0:5], 22050 * 4, 22050, is_train=False)
    test_loader = DataLoader(
        test_set,
        batch_size=2,
        num_workers=2,
        pin_memory=True,
        collate_fn=test_set.collate_augment,
        shuffle=True,
    )
    for i, (embed, class_id, salience, filename) in enumerate(test_loader):
        print(embed.shape)
        plt.imshow(embed[0][0], aspect="auto")
        plt.show()
        break
