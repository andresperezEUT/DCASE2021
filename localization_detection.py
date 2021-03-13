"""
localization_detection.py
"""
import csv
import os
import tempfile

import numpy as np
from utils import doa, diffuseness, circmedian, Event
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matlab import engine
# future = engine.start_matlab(background=True)
# eng = future.result()
# print(eng.sqrt(4.))

def ld_particle(stft, eng, diff_th, K_th, min_event_length, V_azi, V_ele, in_sd, in_sdn, init_birth, in_cp, num_particles, debug_plot=False, metadata_file_path=None):
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
    doa_masked = np.empty((2, K, N))
    for k in range(K):
        for n in range(N):
            if diff_mask[k, n]:
                doa_masked[:, k, n] = DOA[:, k, n]
            else:
                doa_masked[:, k, n] = np.nan

    # # decimate DOA in time
    # DOA_decimated = np.empty((2, K, N // 2))  # todo fix number
    # for n in range(N // 2):
    #     # todo fix numbers depending on decimation factor
    #     # todo: nanmean but circular!!!
    #     meanvalue = np.nanmean([doa_masked[:, :, n * 2], doa_masked[:, :, n * 2 - 1]], axis=0)
    #     meanvalue2 = np.mean([doa_masked[:, :, n * 2], doa_masked[:, :, n * 2 - 1]], axis=0)
    #     # DOA_decimated[:, :, n] = meanvalue
    #     DOA_decimated[:, :, n] = meanvalue2
    #     # if np.any(~np.isnan(meanvalue)):
    #     #     pass
    # M, K, N = DOA_decimated.shape
    #
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

    # if debug_plot:
    #     plt.figure()
    #     # All estimates
    #     for n in range(N):
    #         if len(azis[n]) > 0:
    #             a = np.mod(azis[n] * 180 / np.pi, 360)
    #             plt.scatter(np.ones(len(a)) * n, a, marker='x', edgecolors='b')
    #     # Circmedian
    #     for n in range(N):
    #         if len(azis[n]) > 0:
    #             a = np.mod(azis[n] * 180 / np.pi, 360)
    #             plt.scatter(n, np.mod(circmedian(a, 'deg'), 360), facecolors='none', edgecolors='k')
    #
    #     # circmean and std
    #     plt.figure()
    #     for n in range(N):
    #         if len(azis[n]) > 0:
    #             a = np.mod(azis[n] * 180 / np.pi, 360)
    #             plt.errorbar(n, scipy.stats.circmean(a, high=360, low=0), yerr= scipy.stats.circstd(a, high=360, low=0))
    #             plt.scatter(n, np.mod(circmedian(a, 'deg'), 360), facecolors='none', edgecolors='k')
    #
    #
    #     # boxplot
    #     import seaborn as sns
    #     a = []
    #     for n in range(N):
    #         if len(azis[n]) > 0:
    #             a.append(np.mod(azis[n] * 180 / np.pi, 360))
    #         else:
    #             a.append([])
    #     plt.figure()
    #     sns.boxplot(data=a)
    #
    #     # number of single-source bins in frequency for each n
    #     plt.figure()
    #     plt.grid()
    #     for n in range(N):
    #         if len(azis[n]) > 0:
    #             plt.scatter(n, len(azis[n]), marker='x',  edgecolors='b')

    # TODO: separate frames with two overlapping sources

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
    # this_file_path = os.path.dirname(os.path.abspath(__file__))
    # matlab_path = this_file_path + '/../multiple-target-tracking-master'
    # eng.addpath(matlab_path)
    eng.func_tracking(csv_file_path, float(V_azi), float(V_ele), float(in_sd),
                      float(in_sdn), init_birth, in_cp, float(num_particles), nargout=0)

    # Load output matlab file
    output = loadmat(output_file_path)
    output_data = output['tracks'][0]
    num_events = output_data.size
    # each element of output_data is a different event
    # order of stored data is [time][[azis][eles][std_azis][std_eles]]

    # convert output data into Events
    min_length = min_event_length
    event_list = []
    for n in range(num_events):

        frames = (output_data[n][0][0] / 0.1).astype(int)  # frame numbers

        # sometimes there are repeated frames; clean them
        diff = frames[1:] - frames[:-1]
        frames = np.insert(frames[1:][diff != 0], 0, frames[0])

        if len(frames) > min_length:
            azis = output_data[n][1][0] * np.pi / 180.  # in rads
            azis = [a - (2*np.pi) if a > np.pi else a for a in azis] # adjust range to [-pi, pi]
            eles = (90 - output_data[n][1][1]) * np.pi / 180.  # in rads, incl2ele
            event_list.append(Event(-1, -1, frames, azis, eles))


    def trim_event(e):

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

        return Event(-1, -1, np.asarray(new_frames), np.asarray(new_azis), np.asarray(new_eles))

    trimmed_event_list = []
    for e in event_list:
        trimmed_event_list.append(trim_event(e))
    event_list = trimmed_event_list

    def interpolate_event(e):

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

        return Event(-1, -1, np.asarray(new_frames), np.asarray(new_azis), np.asarray(new_eles))

    interpolated_event_list = []
    for e in event_list:
        interpolated_event_list.append(interpolate_event(e))
    event_list = interpolated_event_list

    # TODO PARAMETRIZE
    def decimate_event(e):
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
        return Event(-1, -1, np.asarray(new_frames), np.asarray(new_azis), np.asarray(new_eles))

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

    if debug_plot:
        # # plot doa estimates and particle trajectories
        # plt.figure()
        # plt.grid()
        # # framewise estimates
        # est_csv = np.loadtxt(open(csv_file_path, "rb"), delimiter=",")
        # t = est_csv[:, 0] * 10
        # a = est_csv[:, 1]
        # e = est_csv[:, 2]
        # plt.scatter(t, a, marker='x', edgecolors='b')
        # # particle filter
        # for e_idx, e in enumerate(event_list):
        #     azis = np.asarray(e.get_azis()) * 180 / np.pi
        #     azis = [a + (360) if a < 0 else a for a in azis] # adjust range to [-pi, pi]
        #     plt.plot(e.get_frames(), azis, marker='.', color='chartreuse')

        #  PLOT # todo check elevation/inclination
        plt.figure()
        title_string = str(V_azi) + '_' + str(V_ele) + '_' + str(in_sd) + '_' + str(in_sdn) + '_' + str(
            init_birth) + '_' + str(in_cp) + '_' + str(num_particles)
        plt.title(title_string)
        plt.grid()

        # framewise estimates
        est_csv = np.loadtxt(open(csv_file_path, "rb"), delimiter=",")
        t = est_csv[:, 0] * 10 / 2 # TODO: ADAPTIVE DECIMATION
        a = est_csv[:, 1]
        e = est_csv[:, 2]
        plt.scatter(t, a, marker='x', edgecolors='b')

        # groundtruth
        gt_csv = np.loadtxt(open(metadata_file_path, "rb"), delimiter=",")
        t = gt_csv[:, 0]
        a = np.mod(gt_csv[:, 3], 360)
        e = gt_csv[:, 4]
        plt.scatter(t, a, facecolors='none', edgecolors='r')

        # particle filter
        for e_idx, e in enumerate(event_list):
            azis = e.get_azis() * 180 / np.pi
            azis = [a + 360 if a < 0 else a for a in azis]  # adjust range to [-pi, pi]

            plt.plot(e.get_frames(), azis, marker='.', markersize=1, color='chartreuse')

    return event_list
