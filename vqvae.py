import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks_1 = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.BatchNorm2d(channel),
            ]

            blocks_2 = [
                nn.Conv2d(in_channel, channel // 2, 2, stride=2, padding=0),
                nn.BatchNorm2d(channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 2, stride=2, padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.BatchNorm2d(channel),
            ]

            blocks_3 = [
                nn.Conv2d(in_channel, channel // 2, 6, stride=2, padding=2),
                nn.BatchNorm2d(channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 6, stride=2, padding=2),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.BatchNorm2d(channel),
            ]

            blocks_4 = [
                nn.Conv2d(in_channel, channel // 2, 8, stride=2, padding=3),
                nn.BatchNorm2d(channel // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 8, stride=2, padding=3),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.BatchNorm2d(channel),
            ]

        for i in range(n_res_block):
            blocks_1.append(ResBlock(channel, n_res_channel))
            blocks_2.append(ResBlock(channel, n_res_channel))
            blocks_3.append(ResBlock(channel, n_res_channel))
            blocks_4.append(ResBlock(channel, n_res_channel))

        blocks_1.append(nn.ReLU(inplace=True))
        blocks_2.append(nn.ReLU(inplace=True))
        blocks_3.append(nn.ReLU(inplace=True))
        blocks_4.append(nn.ReLU(inplace=True))

        self.blocks_1 = nn.Sequential(*blocks_1)
        self.blocks_2 = nn.Sequential(*blocks_2)
        self.blocks_3 = nn.Sequential(*blocks_3)
        self.blocks_4 = nn.Sequential(*blocks_4)

    def forward(self, input):
        return (
                self.blocks_1(input)
                + self.blocks_2(input)
                + self.blocks_3(input)
                + self.blocks_4(input)
        )


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
            self,
            in_channel=1,  # for mel-spec.
            channel=256,
            n_res_block=3,
            n_res_channel=64,
            embed_dim=128,
            n_embed=1024,
            decay=0.99,
            n_classes=None,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed, decay=decay)

        if n_classes is not None:
            self.label_classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(embed_dim, n_classes),
            )
            self.classifier_loss = torch.nn.CrossEntropyLoss()

        self.dec = Decoder(
            embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input, labels=None):
        quant_b, diff, id_b = self.encode(input)
        dec = self.decode(quant_b)

        if labels is not None:
            pred_logits = self.label_classifier(quant_b)
            pred_loss = self.classifier_loss(pred_logits, labels)
            return dec, diff, pred_loss

        return dec, diff, torch.zeros(1, device=input.device)

    def encode(self, input):
        with torch.cuda.amp.autocast():
            enc_b = self.enc_b(input)
            quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)

        # Recast quant_b to fp32 for quantize_b
        quant_b = quant_b.float()
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_b, diff_b, id_b

    def decode(self, quant_b):
        with torch.cuda.amp.autocast():
            dec = self.dec(quant_b)

        dec = dec.float()
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_b)

        return dec


def melspec_train():
    import audio2mel
    from datasets import get_dataset_filelist
    from torch.utils.data import DataLoader
    from torchinfo import summary

    train_file_list, _, _ = get_dataset_filelist(0, 0)

    train_set = audio2mel.Audio2Mel(
        train_file_list[0:5], 22050 * 4, 1024, 80, 256, 22050, 0, 8000
    )

    mel, class_id, _, filename = train_set[0]

    loader = DataLoader(train_set, batch_size=2, sampler=None, num_workers=0)

    model = VQVAE(n_classes=7)
    model = model.to('cuda')

    for i, batch in enumerate(loader):
        mel, class_id, _, filename = batch
        mel = mel.to('cuda')
        class_id = class_id.to('cuda')
        out, latent_loss, class_loss = model(mel, labels=class_id)
        print(out.shape)
        if i == 5:
            break
    summary(model, input_size=mel.shape)


def cembed_train():
    from torch.utils.data import DataLoader
    import datasets
    from audio2cembed import Audio2CEmbed
    from torchinfo import summary

    train_file_list, valid_file_list, test_file_list = datasets.get_dataset_filelist_from_csv('dataset_splits.csv')
    train_set = Audio2CEmbed(train_file_list[0:5], 22050 * 4, 22050, is_train=True)
    train_loader = DataLoader(train_set,
                              batch_size=2,
                              num_workers=2,
                              pin_memory=True,
                              collate_fn=train_set.collate_augment,
                              shuffle=True)

    model = VQVAE(n_classes=7)
    model = model.to('cuda')

    for i, (embed, label, class_id, salience, filename) in enumerate(train_loader):
        embed = embed.to('cuda')
        class_id = class_id.to('cuda')
        out, latent_loss, class_loss = model(embed, labels=class_id)
        print(out.shape)
        if i == 5:
            break
    summary(model, input_size=embed.shape)


if __name__ == '__main__':
    # melspec_train()
    cembed_train()
