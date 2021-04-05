"""
localization_detection.py
"""
import csv
import os
import tempfile
import numpy as np
from evaluation_metrics import distance_between_gt_pred
from utils import doa, diffuseness, circmedian, Event, compute_spectrogram
from scipy.io import loadmat
import matplotlib.pyplot as plt
import soundfile as sf
import config as conf


########################################################################
# High level functions

def localize_detect(parameters, audio_file_name):

    # If there is precomputed spectrogram, go for it
    stft = None
    file_name = os.path.split(os.path.splitext(audio_file_name)[0])[-1]
    stft_file_path = os.path.join(conf.stft_folder_path, file_name+'.npy')
    if os.path.exists(stft_file_path):
        stft = np.load(stft_file_path, allow_pickle=False)
    # Otherwise, compute it
    else:
        audio_file_path = os.path.join(conf.data_folder_path, audio_file_name)
        b_format, sr = sf.read(audio_file_path)
        stft_method_args = ['hann', conf.window_size, conf.window_overlap, conf.nfft]
        stft = compute_spectrogram(b_format, sr, *stft_method_args)

    # ############################################
    # Localization and detection analysis: from stft to event_list
    ld_method_args = [parameters['diff_th'],
                      parameters['K_th'],
                      parameters['min_event_length'],
                      parameters['V_azi'],
                      # parameters['V_ele'],
                      parameters['V_azi']/2, # restrict dimensionality on vertical velocity
                      parameters['in_sd'],
                      parameters['in_sdn'],
                      parameters['init_birth'],
                      parameters['in_cp'],
                      parameters['num_particles'],
                      ]
    est_event_list = ld_particle(stft, conf.eng, *ld_method_args)
    return est_event_list

def get_groundtruth(audio_file_name):
    # ############################################
    pred_file_name = audio_file_name.split('/')[-1].split('.')[0] + '.csv'
    pred_file_path = os.path.join(conf.csv_file_path, pred_file_name)
    gt_event_list = parse_annotations(pred_file_path)
    return gt_event_list

def get_evaluation_metrics(est_event_list, gt_event_list, parameters, plot=False):
    doa_error, frame_recall = evaluate_doa(est_event_list, gt_event_list, plot=plot)
    _, event_precision, event_recall = \
        assign_events(est_event_list, gt_event_list, parameters['event_similarity_th'], plot=plot)
    return event_precision, event_recall, doa_error, frame_recall



########################################################################

def parse_annotations(csv_path):
    """
    parse annotation file and return event_list
    :param csv_path: path to csv file
    :return: event_list
    """

    # Open file
    annotation_file = np.loadtxt(open(csv_path, "rb"), delimiter=",")

    ############################################
    # Delimite events

    event_list = []

    # Iterate over the file and fill the info
    for row in annotation_file:
        frame = int(row[0])
        classID = int(row[1])
        eventNumber = int(row[2])
        azi = row[3] * np.pi / 180 # gt is in degrees, but Event likes rads
        ele = row[4] * np.pi / 180

        # Check if event with same eventNumber exists
        event_exists = False
        existing_event_idx = None # this will hold eventNumber, in case
        for event_idx, event in enumerate(event_list):
            if event.get_eventNumber() == eventNumber:
                event_exists = True
                existing_event_idx = event_idx
                break
        # if it exists, append new data
        if event_exists:
            event = event_list[existing_event_idx]
            event.add_frame(frame)
            event.add_azi(azi)
            event.add_ele(ele)

        # if it doesn't exist, create a new one
        else:
            event = Event(classID, eventNumber, [frame], [azi], [ele])
            event_list.append(event)

    return event_list

def trim_event(e):

    eventNumber = e.get_eventNumber()
    frames = e.get_frames()
    azis = e.get_azis()
    eles = e.get_eles()

    diff = frames[1:] - frames[:-1]
    # large diffs tend to be at the end, so just discard everything after the peak
    peak = np.argwhere(diff>40) # TODO ACHTUNG: HARDCODED VALUE
    if peak.size>0:
        # until the peak
        peak_idx = peak[0][0]
        new_frames = frames[:peak_idx+1]
        new_azis = azis[:peak_idx+1]
        new_eles = eles[:peak_idx+1]
    else:
        # just copy
        new_frames = frames
        new_azis = azis
        new_eles = eles

    return Event(-1, eventNumber, np.asarray(new_frames), np.asarray(new_azis), np.asarray(new_eles))


def interpolate_event(e):

    eventNumber = e.get_eventNumber()
    frames = e.get_frames()
    azis = e.get_azis()
    eles = e.get_eles()

    new_frames = []
    new_azis = []
    new_eles = []

    frame_dist = frames[1:] - frames[:-1]
    for fd_idx, fd in enumerate(frame_dist):
        if fd == 1:
            # contiguous, set next
            new_frames.append(frames[fd_idx])
            new_azis.append(azis[fd_idx])
            new_eles.append(eles[fd_idx])
        else:
            start = frames[fd_idx]
            end = frames[fd_idx+1]
            new_frames.extend(np.arange(start, end, 1).tolist())
            new_azis.extend(np.linspace(azis[fd_idx], azis[fd_idx+1], fd).tolist())
            new_eles.extend(np.linspace(eles[fd_idx], eles[fd_idx+1], fd).tolist())

    return Event(-1, eventNumber, np.asarray(new_frames), np.asarray(new_azis), np.asarray(new_eles))

def decimate_event(e):
    eventNumber = e.get_eventNumber()
    frames = e.get_frames()
    azis = e.get_azis()
    eles = e.get_eles()

    new_frames = []
    new_azis = []
    new_eles = []

    for f_idx, f in enumerate(frames):
        if f%2==1: # only odd
            new_frames.append(f//2)
            new_azis.append(azis[f_idx])
            new_eles.append(eles[f_idx])
    return Event(-1, eventNumber, np.asarray(new_frames), np.asarray(new_azis), np.asarray(new_eles))

def ld_particle(stft, eng, diff_th, K_th, min_event_length, V_azi, V_ele, in_sd, in_sdn, init_birth, in_cp, num_particles):
    """
    find single-source tf-bins, and then feed them into the particle tracker
    :param stft:
    :param diff_th:
    :return:

    """

    # decimate in frequency
    M, K, N = stft.shape
    stft = stft[:, :K // 2, :]
    M, K, N = stft.shape

    # parametric analysis
    DOA = doa(stft)  # Direction of arrival
    diff = diffuseness(stft, dt=2)  # Diffuseness
    diff_mask = diff <= diff_th
    diff_mask[0] = False # manually set artifacts of low diffuseness in the low end spectrum

    # create masked doa with nans
    doa_masked = np.copy(DOA)
    doa_masked[0][~diff_mask] = np.nan
    doa_masked[1][~diff_mask] = np.nan

    DOA_decimated = doa_masked

    # Create lists of azis and eles for each output frame size
    # Filter out spureous candidates
    azis = [[] for n in range(N)]
    eles = [[] for n in range(N)]
    for n in range(N):
        a = DOA_decimated[0, :, n]
        e = DOA_decimated[1, :, n]
        azis_filtered = a[~np.isnan(a)]
        if len(azis_filtered) > K_th:
            azis[n] = azis_filtered
            eles[n] = e[~np.isnan(e)]

    # Save into temp file
    fo = tempfile.NamedTemporaryFile()
    csv_file_path = fo.name + '.csv'
    output_file_path = (os.path.splitext(csv_file_path)[0]) + '.mat'

    with open(csv_file_path, 'a') as csvfile:
        writer = csv.writer(csvfile)
        for n in range(len(azis)):
            if len(azis[n]) > 0:  # if not empty, write
                # time = n * seconds_per_frame
                time = n * 0.1

                # TODO: IQR TEST
                azi = np.mod(circmedian(azis[n]) * 180 / np.pi, 360)  # csv needs degrees, range 0..360
                ele = 90 - (np.median(eles[n]) * 180 / np.pi)  # csv needs degrees
                writer.writerow([time, azi, ele])


    # Call Matlab
    try:
        eng.func_tracking(csv_file_path, float(V_azi), float(V_ele), float(in_sd),
                          float(in_sdn), init_birth, in_cp, float(num_particles), nargout=0)
    except:
        return []


    # Load output matlab file
    output = loadmat(output_file_path)
    output_data = output['tracks'][0]
    num_events = output_data.size
    # each element of output_data is a different event
    # order of stored data is [time][[azis][eles][std_azis][std_eles]]

    # convert output data into Events
    min_length = min_event_length
    event_list = []
    event_count = 0
    for n in range(num_events):

        frames = (output_data[n][0][0] / 0.1).astype(int)  # frame numbers

        # sometimes there are repeated frames; clean them
        diff = frames[1:] - frames[:-1]
        frames = np.insert(frames[1:][diff != 0], 0, frames[0])

        if len(frames) > min_length:
            azis = output_data[n][1][0] * np.pi / 180.  # in rads
            azis = [a - (2*np.pi) if a > np.pi else a for a in azis] # adjust range to [-pi, pi]
            eles = (90 - output_data[n][1][1]) * np.pi / 180.  # in rads, incl2ele
            event_list.append(Event(-1, event_count, frames, azis, eles))
            event_count = event_count + 1

    trimmed_event_list = []
    for e in event_list:
        trimmed_event_list.append(trim_event(e))
    event_list = trimmed_event_list

    interpolated_event_list = []
    for e in event_list:
        interpolated_event_list.append(interpolate_event(e))
    event_list = interpolated_event_list

    # Decimate list
    decimated_event_list = []
    for e in event_list:
        decimated_event_list.append(decimate_event(e))
    event_list = decimated_event_list

    # Check that all events have data
    filtered_event_list = []
    for e in event_list:
        if len(e.get_frames()) > 0:
            filtered_event_list.append(e)
    event_list = filtered_event_list

    return event_list



def evaluate_doa(pred_event_list, gt_event_list, plot=False):
    """
    Given an estimated event list and the corresponding groundtruth event list,
    compute the doa score (doa error and frame recall) based in DCASE2019 DOA metrics (class-independent)
    :param pred_event_list: predicted event list
    :param gt_event_list:  annotated event list
    :return: tuple (doa error, number of events estimation distance)
    """

    def get_doas_at_frame(event_list, frame):
        doas = []
        for e in event_list:
            frames = e.get_frames()
            azis = e.get_azis()
            eles = e.get_eles()
            if frame in frames:
                if type(frames) == np.ndarray:
                    frames = list(frames)
                idx = frames.index(frame)
                doas.append([azis[idx], eles[idx]])
        return doas

    nb_frames = 600
    total_distance = 0

    for n in range(nb_frames):
        gt_array = get_doas_at_frame(gt_event_list, n)
        pred_array = get_doas_at_frame(pred_event_list, n)
        if len(gt_array) > 0 and len(pred_array) > 0:
            distance = distance_between_gt_pred(np.asarray(gt_array), np.asarray(pred_array))
            total_distance += distance

    def get_total_num_doas(event_list):
        count = 0
        for e in event_list:
            count += len(e.get_frames())
        return count

    total_num_predicted_doas = get_total_num_doas(pred_event_list)

    doa_error = total_distance / total_num_predicted_doas

    ## estimation distance

    def get_num_doas_per_frame(event_list):
        doas_count = np.zeros(600) # todo parameter
        for e in event_list:
            frames = e.get_frames()
            for frame in frames:
                doas_count[frame] += 1
        return doas_count

    gt_num_doas = get_num_doas_per_frame(gt_event_list)
    pred_num_doas = get_num_doas_per_frame(pred_event_list)
    true_positives = np.sum(pred_num_doas == gt_num_doas)
    false_negatives = np.sum(pred_num_doas < gt_num_doas)
    false_positives = np.sum(pred_num_doas > gt_num_doas)
    recall = (true_positives + false_positives) / (true_positives + false_positives + false_negatives)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(pred_num_doas, label='pred')
    # plt.plot(gt_num_doas, label='gt')
    # plt.grid()
    # plt.legend()

    return doa_error, recall


def assign_events(pred_event_list, gt_event_list, similiarity_th=0.5, plot=False):

    def common_elements(list1, list2):
        return sorted(list(set(list1).intersection(list2)))

    gt_len, pred_len = len(gt_event_list), len(pred_event_list)
    time_similarity_matrix = np.zeros((gt_len, pred_len))
    space_similarity_matrix = np.zeros((gt_len, pred_len))

    # Slow implementation

    for gt_event_idx, gt_event in enumerate(gt_event_list):
        for pred_event_idx, pred_event in enumerate(pred_event_list):
            # check time coincidence
            gt_frames = gt_event.get_frames()
            pred_frames = pred_event.get_frames()
            common_frames = common_elements(gt_frames, pred_frames)
            time_similarity_matrix[gt_event_idx, pred_event_idx] = \
                min( len(common_frames)/len(gt_frames), len(common_frames)/len(pred_frames) )

            # check spatial coincidence - time-restricted
            # only if time similarity is not zero
            if time_similarity_matrix[gt_event_idx, pred_event_idx] > 0:
                for frame in common_frames:
                    index_of_frame_gt = np.where(gt_frames==frame)[0][0]
                    azi_gt = gt_event.get_azis()[index_of_frame_gt]
                    ele_gt = gt_event.get_eles()[index_of_frame_gt]
                    pos_gt = np.asarray([[azi_gt, ele_gt]])

                    index_of_frame_pred = np.where(pred_frames==frame)[0][0]
                    azi_pred = pred_event.get_azis()[index_of_frame_pred]
                    ele_pred = pred_event.get_eles()[index_of_frame_pred]
                    pos_pred = np.asarray([[azi_pred, ele_pred]])

                    # normalized inverse of distance: 1 is coincidence, 0 is maximum distance (pi)
                    space_similarity_matrix[gt_event_idx, pred_event_idx] = \
                        np.abs(np.pi - distance_between_gt_pred(pos_gt, pos_pred)) / np.pi

    # get total similarity and find most likely (highest scoring) pred event for each gt event
    total_similarity_matrix = time_similarity_matrix * space_similarity_matrix
    best_pred_events = np.empty(gt_len) # each element correspond to a best matching gt_ordered index
    best_pred_events[:] = np.nan
    best_pred_events_score = np.empty(gt_len)
    for gt_event_idx in range(gt_len):
        row = total_similarity_matrix[gt_event_idx,:]
        best_candidate = np.where(row == row.max())[0][0]
        best_candidate_score = row[best_candidate]
        if best_candidate_score >= similiarity_th:
            best_pred_events[gt_event_idx] = best_candidate
            best_pred_events_score[gt_event_idx] = best_candidate_score
        # each gt event must map to one different pred event.
        # if there is already a same candidate, keep the more likely and remove the other
        # a nan will be interpreted as a false negative (missed event)
        if best_candidate in best_pred_events[:gt_event_idx]:
            other_candidate_idx = np.where(best_candidate == best_pred_events)[0][0]
            best_candidate_score = total_similarity_matrix[gt_event_idx,best_candidate]
            other_candidate_score = total_similarity_matrix[other_candidate_idx,best_candidate]
            if  best_candidate_score > other_candidate_score:
                best_pred_events[other_candidate_idx] = np.nan
            else:
                best_pred_events[gt_event_idx] = np.nan
    # print(best_pred_events)

    #####################################
    # Assignment evaluation
    # Categories:
    #   - true positives: events correctly detected
    #   - false positives: events incorrectly detected (present in prediction, but not in groundtruth)
    #   - false negatives: events incorrectly not detected (present in groundtruth but not in prediction)
    # Metrics:
    #   - Precision: events correctly detected vs. all events detected
    #   - Recall: events correctly detected vs. all events in groundtruth

    # number of individual estimated events: number of non-nan entries in best_pred_events array
    precision = len(np.where(~np.isnan(best_pred_events))[0]) / pred_len
    recall = len(np.where(~np.isnan(best_pred_events))[0]) / gt_len

    # PLOT
    if plot:
        def highlight_cell(x, y, ax=None, **kwargs):
            rect = plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=False, **kwargs)
            ax = ax or plt.gca()
            ax.add_patch(rect)
            return rect

        plt.figure()
        plt.subplot(311)
        plt.title("time similarity")
        plt.imshow(time_similarity_matrix)
        plt.subplot(312)
        plt.title("space similarity")
        plt.imshow(space_similarity_matrix)
        plt.subplot(313)
        plt.title("total similarity")
        plt.imshow(total_similarity_matrix)
        for idx in range(gt_len):
            highlight_cell(best_pred_events[idx], idx, color="red", linewidth=1)

    # row_ind, col_ind = linear_sum_assignment(cost_mat)
    # cost = cost_mat[row_ind, col_ind].sum()
    return best_pred_events, precision, recall


