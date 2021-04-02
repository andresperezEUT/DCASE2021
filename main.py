"""
main.py
"""
from localization_detection import ld_particle, evaluate_doa, assign_events
from localization_detection import parse_annotations
import time, datetime
import os
import soundfile as sf
from utils import compute_spectrogram, get_mono_audio_from_event, plot_results
import sys

from matlab import engine
eng = engine.start_matlab()
this_file_path = os.path.dirname(os.path.abspath(__file__))
# matlab_path = this_file_path + '/../multiple-target-tracking-master'
matlab_path = '/home/pans/source/DCASE2021/multiple-target-tracking-master'
eng.addpath(matlab_path)


data_folder_path = '/home/pans/datasets/DCASE2021/foa_dev/dev-train'
csv_file_path = '/home/pans/datasets/DCASE2021/metadata_dev/dev-train'
audio_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(data_folder_path) for f in fn]

fs = 24000
label_hop_len_s = 0.1 # todo check that it still holds
window_size = 2400
window_overlap = 1200
nfft = 2400

plot = True

################################################
# PARAMETERS
diff_th = 0.1 # [0, 1] linear
K_th = 10 # [1, 25] linear
min_event_length = 10 # [1, 25] linear
V_azi  = 2 # [0.1, 10] log
V_ele = 1  # [0.1, 10] log
in_sd = 5  # [0, 50] linear
in_sdn = 20  # [0, 50] linear
init_birth = 0.25  # [0, 1] linear
in_cp = 0.25  # [0, 1] linear
num_particles = 30 # [10, 100]
################################################

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'short':
        audio_files = audio_files[:10]
        # audio_files = [audio_files[1]]

    start_time = time.time()
    print('                                              ')
    print('-------------- PROCESSING FILES --------------')
    print('Folder path: ' + data_folder_path)

    for audio_file_idx, audio_file_name in enumerate(audio_files):

        st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
        print("{}: {}, {}".format(audio_file_idx, st, audio_file_name))

        # Open file
        audio_file_path = os.path.join(data_folder_path, audio_file_name)
        b_format, sr = sf.read(audio_file_path)

        # Get spectrogram
        # TODO: move to suitable place
        stft_method_args = ['hann', window_size, window_overlap, nfft]
        stft = compute_spectrogram(b_format, sr, *stft_method_args)

        # ############################################
        # Localization and detection analysis: from stft to event_list
        ld_method_string = 'ld_particle'
        ld_method = locals()[ld_method_string]
        ld_method_args = [0.1, 10, 10, 2, 1, 5, 20, 0.25, 0.25, 30]
        est_event_list = ld_method(stft, eng, *ld_method_args)

        # ############################################
        pred_file_name = audio_file_name.split('/')[-1].split('.')[0] + '.csv'
        pred_file_path = os.path.join(csv_file_path, pred_file_name)
        gt_event_list = parse_annotations(pred_file_path)

        ############################################
        # compute DOA evaluation metrics
        # doa_error, recall = evaluate_doa(est_event_list, gt_event_list)
        # estimation distance
        similarity_th = 0.3
        assignment = assign_events(est_event_list, gt_event_list, similarity_th)
        # plot
        plot_results(est_event_list, gt_event_list, pred_file_name, assign=assignment)
        if plot:
            import matplotlib.pyplot as plt
            plt.show()

    print('-------------- PROCESSING FINISHED --------------')
    print('                                                 ')