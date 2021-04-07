"""
main.py
- Plot localiization and detection results
- Extract mono audios from given configuration
"""

from localization_detection import ld_particle, evaluate_doa, assign_events
from localization_detection import localize_detect, get_groundtruth, get_evaluation_metrics
import time, datetime
import os
import soundfile as sf
from utils import compute_spectrogram, get_mono_audio_from_event, plot_results, get_class_name_dict, create_folder
import sys
import numpy as np
import config as conf


################################################
# Config

plot = True
write = False
output_dataset_name = 'test_event_dataset'

if write:
    # Create output folder structure
    class_name_dict = get_class_name_dict()
    output_path = os.path.join('/home/pans/datasets/DCASE2021/generated', output_dataset_name)
    create_folder(output_path)
    for class_name in class_name_dict.values():
        folder = os.path.join(output_path, class_name)
        create_folder(folder)
    # Helper variables
    occurrences_per_class = np.zeros(conf.num_classes, dtype=int)

audio_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(conf.data_folder_path) for f in fn]

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
event_similarity_th = 0.3

parameters = {}
parameters['diff_th'] = diff_th
parameters['K_th'] = K_th
parameters['min_event_length'] = min_event_length
parameters['V_azi'] = V_azi
parameters['V_ele'] = V_ele
parameters['in_sd'] = in_sd
parameters['in_sdn'] = in_sdn
parameters['init_birth'] = init_birth
parameters['in_cp'] = in_cp
parameters['num_particles'] = num_particles
parameters['event_similarity_th'] = event_similarity_th




##################################################################
# Main loop
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'short':
        audio_files = conf.short_audio_file_list

    start_time = time.time()
    print('                                              ')
    print('-------------- PROCESSING FILES --------------')

    print('Folder path: ' + conf.data_folder_path)
    ##################################################################
    # Iterate over all files
    for audio_file_idx, audio_file_name in enumerate(audio_files):
        st = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
        print("{}: {}, {}".format(audio_file_idx, st, audio_file_name))

        ##################################################
        # Perform localization and detection, and compute evaluation metrics
        est_event_list = localize_detect(parameters, audio_file_name)
        if len(est_event_list) == 0:
            print('Empty list. Continue')
            continue

        gt_event_list = get_groundtruth(audio_file_name)
        doa_error, frame_recall = evaluate_doa(est_event_list, gt_event_list, plot=plot)
        assignment_array, event_precision, event_recall = \
            assign_events(est_event_list, gt_event_list, parameters['event_similarity_th'], plot=plot)

        print('doa_error:', doa_error)
        print('frame_recall:', frame_recall)
        print('event_precision:', event_precision)
        print('event_recall:', event_recall)

        # ###########################################
        # Extract estimated events. Assign them a class if correctly identified. Save them in training dataset.
        for est_event_idx, est_event in enumerate(est_event_list):
            mono_event = get_mono_audio_from_event(audio_file_name, est_event, conf.fs, conf.frame_length)
            # Obtain classID
            classID = None
            if est_event_idx in assignment_array:
                gt_event_idx = np.where(assignment_array == est_event_idx)[0][0]
                classID = gt_event_list[gt_event_idx].get_classID()
            else:
                classID = conf.undefined_classID

            # Write file
            if write:
                file_name = os.path.split(os.path.splitext(audio_file_name)[0])[-1]
                event_occurrence_idx = occurrences_per_class[classID]
                class_name = class_name_dict[classID]
                output_name = str(event_occurrence_idx) + '_' + file_name + '.wav'
                output_name = os.path.join(output_path, class_name, output_name)
                sf.write(output_name, mono_event, conf.fs)
                # increment counter
                occurrences_per_class[classID] += 1


        ###########################################
        # Plot
        if plot:
            plot_results(est_event_list, gt_event_list, audio_file_name, assign=assignment_array)
            import matplotlib.pyplot as plt
            plt.show()


    print('-------------- PROCESSING FINISHED --------------')
    print('                                                 ')