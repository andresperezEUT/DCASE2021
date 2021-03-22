"""
utils.py
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os
import csv
import librosa


# DATA MANAGEMENT


class Event:
    def __init__(self, classID, eventNumber, frames, azis, eles):
        self._classID = classID
        self._eventNumber = eventNumber
        self._frames = frames # frame indices, in target hopsize units (0.1 s/frame)
        self._azis = azis # in rad, range: [-pi, pi]
        self._eles = eles # in rad, range: [-pi/2, pi/2]

    def get_classID(self):
        return self._classID

    def set_classID(self, classID):
        self._classID = classID

    def get_eventNumber(self):
        return self._eventNumber

    def get_frames(self):
        return self._frames

    def get_azis(self):
        return self._azis

    def get_eles(self):
        return self._eles

    def add_frame(self, frame):
        self._frames.append(frame)

    def add_azi(self, azi):
        self._azis.append(azi)

    def add_ele(self, ele):
        self._eles.append(ele)

    def print(self):
        print(self._classID)
        print(self._eventNumber)
        print(self._frames)
        print(self._azis)
        print(self._eles)

    def export_csv(self, csv_file):
        with open(csv_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            for idx in range(len(self._frames)):
                writer.writerow([self._frames[idx],
                                 self._classID,
                                 self._eventNumber,
                                 self._azis[idx]*180/np.pi,     # csv needs degrees
                                 self._eles[idx]*180/np.pi])    # csv needs degrees


def get_class_name_dict():
    return {
        0: 'alarm',
        1: 'crying_baby',
        2: 'crash',
        3: 'barking_dog',
        4: 'female_scream',
        5: 'female_speech',
        6: 'footsteps',
        7: 'knocking_on_door',
        8: 'male_scream',
        9: 'male_speech',
        10:'ringing_phone',
        11:'piano',
        12:'other' # running engine, burning fire and general classes of NIGENS
    }

def plot_event_list(event_list, name):
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    plt.figure()
    plt.suptitle(name)

    plt.subplot(311)
    for e in event_list:
        frames = e.get_frames()
        classID = e.get_classID()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, np.full(len(frames), classID), marker='.', color=cmap[eventNumber], linestyle='None', markersize=4)
    plt.grid()

    plt.subplot(312)
    for e in event_list:
        frames = e.get_frames()
        azis = e.get_azis()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, azis, marker='.', color=cmap[eventNumber], linestyle='None', markersize=4)
    plt.grid()

    plt.subplot(313)
    for e in event_list:
        frames = e.get_frames()
        eles = e.get_eles()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, eles, marker='.', color=cmap[eventNumber], linestyle='None', markersize=4)
    plt.grid()

    plt.show()

def plot_results(est_event_list, gt_event_list, name):
    cmap = ['b', 'r', 'g', 'y', 'k', 'c', 'm', 'b', 'r', 'g', 'y', 'k', 'c', 'm']
    plt.figure()
    plt.suptitle(name)

    # EST
    plt.subplot(321)
    for e in est_event_list:
        frames = e.get_frames()
        classID = e.get_classID()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, np.full(len(frames), classID), marker='.', linestyle='None', markersize=4)
    plt.grid()

    plt.subplot(323)
    for e in est_event_list:
        frames = e.get_frames()
        azis = e.get_azis()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, azis, marker='.', linestyle='None', markersize=4)
    plt.grid()

    plt.subplot(325)
    for e in est_event_list:
        frames = e.get_frames()
        eles = e.get_eles()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, eles, marker='.', linestyle='None', markersize=4)
    plt.grid()

    # GT
    plt.subplot(322)
    for e in gt_event_list:
        frames = e.get_frames()
        classID = e.get_classID()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, np.full(len(frames), classID), marker='.', linestyle='None', markersize=4)
    plt.grid()

    plt.subplot(324)
    for e in gt_event_list:
        frames = e.get_frames()
        azis = e.get_azis()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, azis, marker='.', linestyle='None', markersize=4)
    plt.grid()

    plt.subplot(326)
    for e in gt_event_list:
        frames = e.get_frames()
        eles = e.get_eles()
        eventNumber = e.get_eventNumber()
        plt.plot(frames, eles, marker='.', linestyle='None', markersize=4)
    plt.grid()

    plt.show()


# SIGNAL

def compute_spectrogram(data, sr, window, window_size, window_overlap, nfft, D=None):

    t, f, stft = scipy.signal.stft(data.T, sr, window=window, nperseg=window_size, noverlap=window_overlap, nfft=nfft)
    stft = stft[:,:-1,:-1] # round shape
    M, K, N = stft.shape
    # TODO: check non-integer cases
    if D is not None:
        dec_stft = np.empty((M, K//D, N), dtype=complex)
        for k in range(K//D):
            dec_stft[:,k,:] = stft[:,k*D,:] # decimate
        stft = dec_stft
    return stft

# STATS

def circmedian(angs, unit='rad'):
    """
    circular median!

    :param angs:
    :param unit:
    :return:
    """
    # from https://github.com/scipy/scipy/issues/6644
    # Radians!
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]
    if unit == 'rad':
        pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    elif unit == 'deg':
        pdists = (pdists +180) % (360.) - 180.
    pdists = np.abs(pdists).sum(1)

    # If angs is odd, take the center value
    if len(angs) % 2 != 0:
        return angs[np.argmin(pdists)]
    # If even, take the mean between the two minimum values
    else:
        index_of_min = np.argmin(pdists)
        min1 = angs[index_of_min]
        # Remove minimum element from array and recompute
        new_pdists = np.delete(pdists, index_of_min)
        new_angs = np.delete(angs, index_of_min)
        min2 = new_angs[np.argmin(new_pdists)]
        if unit == 'rad':
            return scipy.stats.circmean([min1, min2], high=np.pi, low=-np.pi)
        elif unit == 'deg':
            return scipy.stats.circmean([min1, min2], high=180., low=-180.)


# PARAMETRIC SPATIAL AUDIO CODING
# Assuming ACN, SN3D data

c = 346.13  # m/s
p0 = 1.1839  # kg/m3

def intensity_vector(stft):
    P = stft[0] # sound pressure
    U = stft[1:] / (p0 * c) # particle velocity # TODO: it really should be -U
    return np.real(U * np.conjugate(P))

def doa(stft):
    I = intensity_vector(stft)
    return np.asarray(cart2sph(I[2], I[0], I[1]))[:-1]

def energy_density(stft):
    P = stft[0]  # sound pressure
    U = stft[1:] / (p0 * c)  # particle velocity # TODO: it really should be -U
    s1 = np.power(np.linalg.norm(U, axis=0), 2)
    s2 = np.power(abs(P), 2)
    return ((p0 / 2.) * s1) + ((1. / (2 * p0 * np.power(c, 2))) * s2)


def diffuseness(stft, dt=5):

    I = intensity_vector(stft)
    E = energy_density(stft)

    M, K, N = stft.shape
    dif = np.zeros((K, N))

    for n in range(int(dt / 2), int(N - dt / 2)):
        num = np.linalg.norm(np.mean(I[:, :, n:n + dt], axis=(2)), axis=0)
        den = c * np.mean(E[:,n:n+dt], axis=1)
        dif[:,n] = 1 - (num/den)

    # Borders: copy neighbor values
    for n in range(0, int(dt / 2)):
        dif[:, n] = dif[:, int(dt / 2)]

    for n in range(int(N - dt / 2), N):
        dif[:, n] = dif[:, int(N - (dt / 2) - 1)]

    return dif

# From evaluation_metrics.py, baseline 2020
def cart2sph(x, y, z):
    '''
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    '''

    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


# BEAMFORMING

def get_ambisonic_gains(azi, ele):
    """
    ACN, N3D
    :param azi: N vector
    :param ele: N vector
    :return:  4 x N matrix
    """
    N = azi.size
    assert N == ele.size
    return np.asarray([np.ones(N), np.sqrt(3)*np.sin(azi)*np.cos(ele), np.sqrt(3)*np.sin(ele), np.sqrt(3)*np.cos(azi)*np.cos(ele)])

def mono_extractor(b_format, azis=None, eles=None, mode='beam'):
    """
    :param b_format: (frames, channels) IN SN3D
    :param mode: 'beamforming' or 'omni'
    :return:
    """
    frames, channels = b_format.shape

    x = np.zeros(frames)

    if mode == 'beam':
        # MaxRE decoding
        b_format_n3d = b_format * np.asarray([1, np.sqrt(3), np.sqrt(3), np.sqrt(3)])  # N3D
        alpha = np.asarray([0.775, 0.4, 0.4, 0.4])  # MaxRE coefs
        decoding_gains = get_ambisonic_gains(azis, eles)  # N3D
        w = decoding_gains * alpha[:, np.newaxis]
        x = np.sum(b_format_n3d * w.T, axis=1)  # N3D BY N3D

    elif mode == 'omni':
        # Just take the W channel
        x = b_format[:, 0]

    return x


def get_mono_audio_from_event(b_format, event, fs, frame_length):
    """
    :param b_format: (frames, channels) IN SN3D
    :param event:
    :param fs:
    :param frame_length:
    :return:
    """

    frames = event.get_frames()
    w = frame_length  # frame length of the annotations
    samples_per_frame = int(w * fs)
    start_time_samples = int(frames[0] * samples_per_frame)
    end_time_samples = int((frames[-1] + 1) * samples_per_frame)  # add 1 here so we push the duration to the end
    mono_event = None


    azi_frames = event.get_azis()
    ele_frames = event.get_eles()
    # frames to samples; TODO: interpolation would be cool
    num_frames = len(frames)
    num_samples = num_frames * samples_per_frame

    assert (end_time_samples - start_time_samples == num_samples)

    azi_samples = np.zeros(num_samples)
    ele_samples = np.zeros(num_samples)
    for idx in range(num_frames):
        azi_samples[(idx * samples_per_frame):(idx + 1) * samples_per_frame] = azi_frames[idx]
        ele_samples[(idx * samples_per_frame):(idx + 1) * samples_per_frame] = ele_frames[idx]

    mono_event = mono_extractor(b_format[start_time_samples:end_time_samples],
                                azis=azi_samples * np.pi / 180,  # deg2rad
                                eles=ele_samples * np.pi / 180,  # deg2rad
                                )


    # normalize audio to 1
    mono_event /= np.max(np.abs(mono_event))
    return mono_event