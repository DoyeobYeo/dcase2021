import os
import sys
import gc
import random

import numpy as np
import scipy.stats
import librosa

from tqdm import tqdm
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader

try:
    from sklearn.externals import joblib
except:
    import joblib

import dcase2021.libs as libs
from dcase2021.libs import file_list_generator


class Dataset(BaseDataset):
    def __init__(
            self,
            info_list, param
    ):
        '''
        Make a Torch Dataset
        :param info_list: List of [wav filename, index of slice]
        하나의 wav 파일을 읽어서 mel spectrogram 변환을 하게 되면, 2D array가 생성된다.
        mel spectrogram에서 time 축의 특정 index의 spectrum 정보를 반환한다.
        :param param: None
        '''
        self.info_list = info_list

        self.n_mels = param["feature"]["n_mels"]
        self.n_frames = param["feature"]["n_frames"]
        self.n_hop_frames = param["feature"]["n_hop_frames"]
        self.n_fft = param["feature"]["n_fft"]
        self.hop_length = param["feature"]["hop_length"]
        self.power = param["feature"]["power"]

    def __getitem__(self, idx):
        file_name = self.info_list[idx][0]
        slice_num = int(self.info_list[idx][1])
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
        try:
            vectors = vectors[slice_num]
        except:
            print(file_name, slice_num, len(y))
        return vectors.astype('float32')

    def __len__(self):
        return len(self.info_list)


def make_dataloader(
        target_dir, param, batch_size,
        num_workers, prefetch_factor, pin_memory, persistent_workers,
        mode=True, validation_split=0.2
):
    '''
    :param target_dir: (string) 각 기기에 대한 데이터들이 위치한 가장 상위 디텍토리 (ex. './dev_data/fan')
    :param param: (dictionary) yaml 파일에 저장된 여러 파라미터를 로드한 값
    :param batch_size: (integer) 배치 사이즈
    :param mode: (boolean) development 모드일 때 True, evaluation 모드일 때 False
    :return: Pytorch DataLoader
    '''
    dataset_info_list = list()
    files, y_true = file_list_generator(target_dir=target_dir,
                                        section_name="*",
                                        dir_name="train",
                                        mode=mode)

    for file_name in files:
        y, sr = libs.file_load(file_name, mono=True)
        feature_mel = 1 + len(y) // param["feature"]["hop_length"] - param["feature"]["n_frames"] + 1
        feature_mel = feature_mel // param["feature"]["n_hop_frames"]
        slices_idx_list = list(np.arange(feature_mel))
        all_list = [[file_name], slices_idx_list]
        comb = [list(x) for x in np.array(np.meshgrid(*all_list)).T.reshape(-1, len(all_list))]
        dataset_info_list.extend(comb)

    val_idx = int(len(dataset_info_list) * validation_split)
    random.shuffle(dataset_info_list)
    val_list = dataset_info_list[:val_idx]
    train_list = dataset_info_list[val_idx:]

    train_dataset = Dataset(
        train_list,
        param
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_dataset = Dataset(
        val_list,
        param
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # test script for pytorch dataloader
    from dcase2021.libs import yaml_load
    from dcase2021.libs import select_dirs


    # "development": mode == True
    # "evaluation": mode == False
    _mode = True
    _param = yaml_load("../baseline.yaml")
    _dirs = select_dirs(param=_param, mode=_mode)

    # loop of the base directory
    for idx, target_dir in enumerate(_dirs):
        # make datasets and dataloaders for each problem

        train_dataloader = make_dataloader(
            target_dir=target_dir,
            param=_param,
            mode=_mode, batch_size=16
        )
        for k in range(5):
            train_tensor = next(iter(train_dataloader))
            print(train_tensor.shape)
