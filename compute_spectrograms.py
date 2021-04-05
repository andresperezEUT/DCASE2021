"""
Compute and save spectrograms of the dataset, in order to save time for other tasks.
STFT parameters are specified in the config.py file
"""
import datetime
import time

import config as conf
import os
import soundfile as sf
from utils import compute_spectrogram, create_folder
import numpy as np


create_folder(conf.stft_folder_path)

audio_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(conf.data_folder_path) for f in fn]
for af_idx, audio_file_name in enumerate(audio_files):
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
    print("{}: {}, {}".format(af_idx, st, audio_file_name))

    audio_file_path = os.path.join(conf.data_folder_path, audio_file_name)
    b_format, sr = sf.read(audio_file_path)
    stft_method_args = ['hann', conf.window_size, conf.window_overlap, conf.nfft]
    stft = compute_spectrogram(b_format, sr, *stft_method_args)

    file_name = os.path.split(os.path.splitext(audio_file_name)[0])[-1]
    output_file_path = os.path.join(conf.stft_folder_path, file_name)
    np.save(output_file_path, stft, allow_pickle=False)