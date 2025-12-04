import torch
from torch.utils.data import Dataset
import librosa
import re
import numpy as np
import utils
import random

def compute_spectral_centroid(x_wav, sr=16000, n_fft=1024, hop_length=512, win_length=None):
    if win_length is None:
        win_length = n_fft
    window = torch.hann_window(win_length).to(x_wav.device)
    stft = torch.stft(
        x_wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    magnitude = stft.abs()
    freqs = torch.linspace(0, sr // 2, steps=magnitude.shape[0]).to(x_wav.device)
    eps = 1e-10
    centroid = (freqs[:, None] * magnitude).sum(dim=0) / (magnitude.sum(dim=0) + eps)
    return centroid


class ASDDataset(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):  # file_list是所有的音频路径file_list
        self.file_list = file_list
        self.args = args
        "1"
        self.wav2mel = utils.Wave2Mel(sr=args.sr, power=args.power,
                                      n_fft=args.n_fft, n_mels=args.n_mels,
                                      win_length=args.win_length, hop_length=args.hop_length)
        "2"
        self.wav2mel2 = utils.Wave2Mel2(sr=16000, power=2.0,
                                        n_fft=512, n_mels=128,
                                        win_length=512, hop_length=256)
        "3"
        self.wav2spec = utils.Wave2spec(power=None,
                                        n_fft=1024,
                                        win_length=1024, hop_length=512)
        self.load_in_memory = load_in_memory
        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []   #  用到提取频谱的那个函数

    def __getitem__(self, item):  # 索引访问对象init时，getitem运行
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])

        return data_item

    def transform(self, filename):
        filename = filename.replace("\\", "/")
        machine = filename.split('/')[-3]
        id_str = re.findall('id_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine+'-'+id_str]  # 0-41
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)  #  返回的是NumPy 数组音频和采样率

        x = x[: self.args.sr * self.args.secs]
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        energy = x_mel
        centroid = compute_spectral_centroid(x_wav,n_fft=1024, hop_length=512, win_length=None) # 313
        return x_wav, x_mel,centroid, energy,label


    def __len__(self):
        return len(self.file_list)


