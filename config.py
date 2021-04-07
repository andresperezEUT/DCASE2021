"""
Experiment configuration file
"""

import os
from matlab import engine
import numpy as np
from utils import get_class_name_dict

##################################################################
# Paths
data_folder_path = '/home/pans/datasets/DCASE2021/foa_dev/dev-train'
stft_folder_path = '/home/pans/datasets/DCASE2021/stft_foa_dev/dev-train'
csv_file_path = '/home/pans/datasets/DCASE2021/metadata_dev/dev-train'
matlab_path = '/home/pans/source/DCASE2021/multiple-target-tracking-master'

# Audio files
short_audio_file_list = [
    'fold1_room1_mix035.wav',
    'fold1_room2_mix035.wav',
    'fold2_room1_mix035.wav',
    'fold2_room2_mix035.wav',
    'fold3_room1_mix035.wav',
    'fold3_room2_mix035.wav',
    'fold4_room1_mix035.wav',
    'fold4_room2_mix035.wav',
]
short_audio_file_list = [os.path.join(data_folder_path, f) for f in short_audio_file_list]


# Matlab
eng = engine.start_matlab()
this_file_path = os.path.dirname(os.path.abspath(__file__))
eng.addpath(matlab_path)

# Audio analysis parameters
fs = 24000
window_size = 2400
window_overlap = 1200
nfft = 2400
frame_length = 0.1 # each annotation frame corresponds to 100 ms

# Classes
num_classes = 13 # 12 regular classes plus uncategorized class
undefined_classID = num_classes - 1
class_name_dict = get_class_name_dict()
