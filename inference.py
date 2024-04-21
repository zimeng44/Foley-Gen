import argparse
import datetime
import json
import math
import os
import time
from abc import ABC, abstractmethod
from typing import List

import librosa
import soundfile as sf
import torch
from HiFiGanWrapper import HiFiGanWrapper
from numpy import ndarray
from pixelsnail import PixelSNAIL
from torch import Tensor
from tqdm import tqdm
from vqvae import VQVAE

class_id_dict: dict = {
    0: "DogBark",
    1: "Footstep",
    2: "Gunshot",
    3: "Keyboard",
    4: "MovingMotorVehicle",
    5: "Rain",
    6: "SneezeCough",
}


class SoundSynthesisModel(ABC):
    @abstractmethod
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        raise NotImplementedError


class DCASE2023FoleySoundSynthesis:
    def __init__(
        self, number_of_synthesized_sound_per_class: int = 100, batch_size: int = 16
    ) -> None:
        self.number_of_synthesized_sound_per_class: int = (
            number_of_synthesized_sound_per_class
        )
        self.batch_size: int = batch_size
        self.class_id_dict: dict = {
            0: "DogBark",
            1: "Footstep",
            2: "Gunshot",
            3: "Keyboard",
            4: "MovingMotorVehicle",
            5: "Rain",
            6: "SneezeCough",
        }
        self.sr: int = 22050
        self.save_dir: str = "./synthesized"

    def synthesize(self, synthesis_model: SoundSynthesisModel) -> None:
        for sound_class_id in self.class_id_dict:
            sample_number: int = 1
            save_category_dir: str = (
                f"{self.save_dir}/{self.class_id_dict[sound_class_id]}"
            )
            os.makedirs(save_category_dir, exist_ok=True)
            for _ in tqdm(
                range(
                    math.ceil(
                        self.number_of_synthesized_sound_per_class / self.batch_size
                    )
                ),
                desc=f"Synthesizing {self.class_id_dict[sound_class_id]}",
            ):
                synthesized_sound_list: list = synthesis_model.synthesize_sound(
                    sound_class_id, self.batch_size
                )
                for synthesized_sound in synthesized_sound_list:
                    if sample_number <= self.number_of_synthesized_sound_per_class:
                        sf.write(
                            f"{save_category_dir}/{str(sample_number).zfill(4)}.wav",
                            synthesized_sound.astype("float32"),
                            samplerate=self.sr,
                        )
                        sample_number += 1


def load_config(dict_or_json_path) -> dict:
    if isinstance(dict_or_json_path, dict):
        return dict_or_json_path

    elif isinstance(dict_or_json_path, str):
        with open(dict_or_json_path, "r") as f:
            config = json.load(f)

        return config


# ================================================================================================================================================
class BaseLineModel(SoundSynthesisModel):
    def __init__(
        self,
        pixel_snail_checkpoint: str,
        vqvae_checkpoint: str,
        hifigan_checkpoint: str,
        pixel_snail_config: dict,
        vqvae_config: dict,
        hifigan_config: dict,
        sr=22050,
    ) -> None:
        super().__init__()

        self.pixel_snail_config = load_config(pixel_snail_config)
        self.pixel_snail = PixelSNAIL(**self.pixel_snail_config)
        self.pixel_snail.load_state_dict(
            torch.load(pixel_snail_checkpoint, map_location="cpu")["model"]
        )
        self.pixel_snail.cuda()
        self.pixel_snail.eval()
        self.pixel_snail.embedNet = torch.compile(self.pixel_snail.embedNet, dynamic=True)

        self.vqvae_config = load_config(vqvae_config)
        self.vqvae = VQVAE(**self.vqvae_config)
        self.vqvae.load_state_dict(
            torch.load(vqvae_checkpoint, map_location="cpu")["model"]
        )
        self.vqvae.cuda()
        self.vqvae.eval()

        self.hifigan_config = load_config(hifigan_config)
        self.hifi_gan = HiFiGanWrapper(hifigan_checkpoint, self.hifigan_config)

        self.sr = sr

    @torch.autocast("cuda")
    def synthesize_sound(self, class_id: str, number_of_sounds: int) -> List[ndarray]:
        audio_list: List[ndarray] = list()
        class_name = class_id_dict[int(class_id)]

        feature_shape: list = self.pixel_snail.shape  # [288, 75]
        vq_token: Tensor = torch.zeros(
            number_of_sounds, *feature_shape, dtype=torch.int64
        ).to("cuda", non_blocking=True)
        cache = dict()

        label_condition = torch.full([number_of_sounds, 1], int(class_id)).long().to("cuda", non_blocking=True)

        with torch.inference_mode():
            for i in tqdm(range(feature_shape[0]), desc="pixel_snail"):
                for j in range(feature_shape[1]):
                    out, cache = self.pixel_snail(
                        vq_token[:, : i + 1, :],
                        label_condition=label_condition,
                        cache=cache,
                    )
                    prob: Tensor = torch.softmax(out[:, :, i, j], 1)
                    vq_token[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)

            # torch.save(vq_token, f"./synthesized/{class_name}/vq_token.pt")
            pred_mel = self.vqvae.decode_code(vq_token).detach()
            # torch.save(pred_mel, f"./synthesized/{class_name}/pred_cembed.pt")

        with torch.no_grad():
            for j, mel in enumerate(pred_mel):
                audio_list.append(self.hifi_gan.generate_audio_by_hifi_gan(mel))

        if self.sr != self.hifigan_config["sampling_rate"]:
            for i, audio in enumerate(audio_list):
                audio_list[i] = librosa.resample(
                    audio,
                    orig_sr=self.hifigan_config["sampling_rate"],
                    target_sr=self.sr,
                    res_type="soxr_hq",
                )

        return audio_list


# ===============================================================================================================================================
if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vqvae_checkpoint", type=str, default="./checkpoint/vqvae/vqvae_epoch.pt"
    )
    parser.add_argument(
        "--pixelsnail_checkpoint",
        type=str,
        default="./checkpoint/pixelsnail-final/bottom_1400.pt",
    )
    parser.add_argument(
        "--hifigan_checkpoint", type=str, default="./checkpoint/hifigan/g_00935000"
    )
    parser.add_argument(
        "--number_of_synthesized_sound_per_class", type=int, default=100
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--vqvae_config", type=str, default=None)
    parser.add_argument("--pixelsnail_config", type=str, default="./configs/pixelsnail_config_v1.json")
    parser.add_argument("--hifigan_config", type=str, default=None)

    args = parser.parse_args()
    dcase_2023_foley_sound_synthesis = DCASE2023FoleySoundSynthesis(
        args.number_of_synthesized_sound_per_class, args.batch_size
    )

    dcase_2023_foley_sound_synthesis.synthesize(
        synthesis_model=BaseLineModel(
            args.pixelsnail_checkpoint,
            args.vqvae_checkpoint,
            args.hifigan_checkpoint,
            args.pixelsnail_config,
            args.vqvae_config,
            args.hifigan_config,
        )
    )
    print(str(datetime.timedelta(seconds=time.time() - start)))
