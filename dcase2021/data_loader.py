import os
import sys
import gc

import numpy as np
import scipy.stats
import librosa

from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset
try:
    from sklearn.externals import joblib
except:
    import joblib

import dcase2021.libs as libs


class Dataset(BaseDataset):
    def __init__(
            self,
            file_list,
            n_mels=64,
            n_frames=5,
            n_hop_frames=1,
            n_fft=1024,
            hop_length=512,
            power=2.0
    ):
        self.file_list = file_list
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_hop_frames = n_hop_frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        dims = self.n_mels * self.n_frames

        # generate melspectrogram using librosa
        y, sr = libs.file_load(file_name, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                         sr=sr,
                                                         n_fft=self.n_fft,
                                                         hop_length=self.hop_length,
                                                         n_mels=self.n_mels,
                                                         power=self.power)

        # convert melspectrogram to log mel energies
        log_mel_spectrogram = 20.0 / self.power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

        # calculate total vector size
        n_vectors = len(log_mel_spectrogram[0, :]) - self.n_frames + 1
        # skip too short clips
        if n_vectors < 1:
            return np.empty((0, dims))

        # generate feature vectors by concatenating multiframes
        vectors = np.zeros((n_vectors, dims))
        for t in range(self.n_frames):
            vectors[:, self.n_mels * t: self.n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + n_vectors].T

        vectors = vectors[::self.n_hop_frames, :]
        return vectors

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    from dcase2021.libs import yaml_load
    from dcase2021.libs import select_dirs
    from dcase2021.libs import file_list_generator
    from torch.utils.data import DataLoader

    # "development": mode == True
    # "evaluation": mode == False
    mode = True
    param = yaml_load("../baseline.yaml")

    dirs = select_dirs(param=param, mode=mode)
    # print(param)

    # loop of the base directory
    file_list = list()
    for idx, target_dir in enumerate(dirs):
        files, y_true = file_list_generator(target_dir=target_dir,
                                            section_name="*",
                                            dir_name="train",
                                            mode=mode)
        file_list.extend(files)

    train_dataset = Dataset(
        file_list
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True,
        pin_memory=True
    )
    for k in range(5):
        train_tensor = next(iter(train_dataloader))
        print(train_tensor.shape)
        tmp = train_tensor.view(-1)
        print(tmp.shape)

