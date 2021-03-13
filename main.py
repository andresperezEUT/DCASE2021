"""
main.py
"""
from localization_detection import ld_particle
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


data_folder_path = '/home/pans/datasets/DCASE2021/foa_dev'
audio_files = [os.path.join(dp, f) for dp, dn, fn in os.walk('/home/pans/datasets/DCASE2021/foa_dev') for f in fn]

fs = 24000
label_hop_len_s = 0.1 # todo check that it still holds


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'short':
        audio_files = audio_files[:10]

    start_time = time.time()
    print('                                              ')
    print('-------------- PROCESSING FILES --------------')
    print('Folder path: ' + data_folder_path)

    for audio_file_idx, audio_file_name in enumerate(audio_files):

        st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
        print("{}: {}, {}".format(audio_file_idx, st, audio_file_name))

        ############################################
        # Preprocess: prepare file output in case
        # if write:
        #     csv_file_name = (os.path.splitext(audio_file_name)[0]) + '.csv'
        #     csv_file_path = os.path.join(result_folder_path, csv_file_name)
        #     # since we always append to the csv file, make a reset on the file
        #     if os.path.exists(csv_file_path):
        #         # os.remove(csv_file_path)
        #         continue # SKIP EXISTING FILES!

        ############################################
        # Open file
        audio_file_path = os.path.join(data_folder_path, audio_file_name)
        b_format, sr = sf.read(audio_file_path)

        # Get spectrogram
        # TODO: move to suitable place
        stft_method_args = ['hann', 2400, 1200, 2400, None]
        stft = compute_spectrogram(b_format, sr, *stft_method_args)

        # ############################################
        # Localization and detection analysis: from stft to event_list
        ld_method_string = 'ld_particle'
        ld_method = locals()[ld_method_string]
        ld_method_args = [0.1, 10, 10, 2, 1, 5, 20, 0.25, 0.25, 30]
        event_list = ld_method(stft, eng, *ld_method_args)

        ############################################
        # Get monophonic estimates of the event, and predict the classes
        num_events = len(event_list)
        for event_idx in range(num_events):
            event = event_list[event_idx]
            mono_event = get_mono_audio_from_event(b_format, event, fs, label_hop_len_s)

        print(mono_event)

        # Predict
        #     class_method_string = params['class_method']
        #     class_method = locals()[class_method_string]
        #     class_method_args = params['class_method_args']
        #     class_idx = class_method(mono_event, *class_method_args)
        #     # class_idx = class_method(temp_file_name, *class_method_args)
        #     event.set_classID(class_idx)
        #     ############################################
        #     # Postprocessing:
        #     process_event = True
        #     try:
        #         event_filter = params['event_filter_activation']
        #     except:
        #         event_filter = False  # default True, so it works also when no event_filter
        #     if event_filter:
        #         event_filter_method_string = params['event_filter_method']
        #         event_filter_method = locals()[event_filter_method_string]
        #         event_filters_method_args = params['event_filter_method_args']
        #         process_event = event_filter_method(event, *event_filters_method_args)
        #
        #     ############################################
        #     # Generate metadata file from event
        #     if write and process_event:
        #         event.export_csv(csv_file_path)
        #
        # ############################################
        # # Plot results
        plot_results(csv_file_path, params)

    print('-------------- PROCESSING FINISHED --------------')
    print('                                                 ')