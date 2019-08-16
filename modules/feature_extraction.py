
#from numba import vectorize
#from timeit import default_timer as timer

#########--------------------------------------------------------------------
import gc
import librosa
import numpy as np
import torch

__author__ = "Vinay Kumar"
__copyright__ = "copyright 2018, Project SSML"
__maintainer__ = "Vinay Kumar and Ramesh Kunasi"
__status__ = "Research & Development"


def extract_stft(data_paths, sr=44100, size_timeframe=15, size_freqframe=1025, 
                    size_fft=2048, size_hop=512, size_window=2048, type_window='hann'
                ):

    """
    Inputs:
    - data_paths        [type=dict] : A (2x2) dictionary containing the relative paths(type=str) of wavfiles for inputs and labels
    - sr                [type=int]  : sampling rate for stft extraction
    - size_timeframe    [type=int]  : size of spectrogram slice in temporal axis
    - size_freqframe    [type=int]  : size of spectrogram slice in spectral axis
    - size_fft          [type=int]  : FFT point size
    - size_hop          [type=int]  : hop length for FFT calculation
    - size_window       [type=int]  : window size for FFT calculation
    - type_window       [type=str]  : a window specification (see scipy.signal.get_window) eg. 'hann', 'hamm' etc.

    Returns:
    - [type=dict] : a dictionary containing the stacked stft features separately for inputs and labels

    """

    print('STFT feature Extraction: START..............')

    data_feat_stack_dict = {}

    for i in data_paths.keys():
        for j in range(data_paths[i].shape[0]):
            wav_file, _ = librosa.load(data_paths[i][j], sr=sr)         # bottleneck
            feat_abs = np.abs(librosa.stft(y = wav_file, n_fft = size_fft, hop_length = size_hop, win_length = size_window, window = type_window))
            start_idx=0
            print(f'i={i}, j={j}')
            for k in range(int(feat_abs.shape[1]/size_timeframe)):
                if j==0 and k==0:
                    # print('i={}, j={}, k={}'.format(i, j, k))
                    data_feat_stack_dict[i] = feat_abs[:, start_idx:start_idx + size_timeframe]
                    start_idx += size_timeframe
                else:
                    # print('i={}, j={}, k={}'.format(i, j, k))
                    data_feat_stack_dict[i] = np.dstack((data_feat_stack_dict[i], feat_abs[:, start_idx:start_idx + size_timeframe]))
                    start_idx += size_timeframe
            del feat_abs
            del wav_file

    print('STFT feature Extraction: END')

    return data_feat_stack_dict

#########--------------------------------------------------------------------

def extract_stft_cuda(device, data_paths, to_mono=False, sr=44100, size_timeframe=15, size_freqframe=1025, 
                        size_fft=2048, size_hop=512, size_window=2048, type_window='hann'
                    ):
    """
    Inputs:
    - device            [type=torch.device] : 'cpu' or 'cuda'
    - data_paths        [type=dict]         : A (2x2) dictionary containing the relative paths of wavfiles for inputs and labels
    - sr                [type=int]          : sampling rate for stft extraction
    - size_timeframe    [type=int]          : size of spectrogram slice in temporal axis
    - size_freqframe    [type=int]          : size of spectrogram slice in spectral axis
    - size_fft          [type=int]          : FFT point size
    - size_hop          [type=int]          : hop length for FFT calculation
    - size_window       [type=int]          : window size for FFT calculation
    - type_window       [type=str]          : a window specification (see scipy.signal.get_window) eg. 'hann', 'hamm' etc.

    Returns:
    - [type=torch.tensor] : a torch.tensor containing the stacked stft features separately for inputs and labels

    """
    print(f'Converting the wavfiles to mono?: {to_mono}')

    blueprint = torch.zeros(1, 1, size_freqframe, size_timeframe, device=device)

    data_feat_cat_tensor = None
    key_count = 0


    for i in data_paths.keys():
        X_list = []
        
        for j in range(data_paths[i].shape[0]):
            feat_stack_tensor_j = None
            wav_file, _ = librosa.load(data_paths[i][j], sr=sr)         # bottleneck
            
            if to_mono:
                wav_file = librosa.to_mono(y=wav_file)          # converting the stereo channel to mono channel
                gc.collect()

            feat_abs = np.abs(librosa.stft(y=wav_file, n_fft=size_fft, hop_length=size_hop, win_length=size_window, window=type_window))
            # feat_tensor = torch.stft(signal=wav_file, frame_length=size_fft, hop=size_hop, fft_size=size_fft, window=type_window).to(device)
            # feat_abs_tensor = mee_abs_cuda(feat_tensor[:,:,0], feat_tensor[:,:,1])
            feat_abs_tensor = torch.tensor(feat_abs, device=device)

            start_idx=0

            print(f'extract_stft_cuda : i={i}, j={j}')
            
            for k in range(int(feat_abs_tensor.shape[1]/size_timeframe)):
                if k==0:
                    feat_stack_tensor_j = (feat_abs_tensor[:, start_idx:start_idx + size_timeframe]).view_as(blueprint)
                    # print('i={}, j={}, k={}'.format(i, j, k))
                    # data_feat_stack_dict[i] = feat_abs[:, start_idx:start_idx + size_timeframe]
                    start_idx += size_timeframe
                else:
                    feat_stack_tensor_j = torch.cat((feat_stack_tensor_j, (feat_abs_tensor[:, start_idx:start_idx + size_timeframe]).view_as(blueprint)), dim=1)
                    start_idx += size_timeframe
                    # print('i={}, j={}, k={}'.format(i, j, k))
            
            X_list.append(feat_stack_tensor_j)
            
            del wav_file
            del feat_abs
            del feat_abs_tensor
            del feat_stack_tensor_j

        
        feat_stack_tensor = X_list.pop(0).to(torch.device('cpu'))
        
        while len(X_list)>0:
            print(f'len(X_list) = {len(X_list)}')
            feat_stack_tensor = torch.cat((feat_stack_tensor, X_list.pop(0).to('cpu')), dim=1)
            gc.collect()

        if key_count==0:
            # for inputs
            print('inside if-loop key_count = {}'.format(key_count))
            data_feat_cat_tensor = feat_stack_tensor
            del feat_stack_tensor
            gc.collect()
            print('data_feat_cat_tensor.shape = {}'.format(data_feat_cat_tensor.shape))
        elif key_count==1:
            # for labels
            print('inside if-loop key_count = {}'.format(key_count))
            data_feat_cat_tensor = torch.cat((data_feat_cat_tensor, feat_stack_tensor), dim=0)
            del feat_stack_tensor
            gc.collect()
            print('data_feat_cat_tensor.shape = {}'.format(data_feat_cat_tensor.shape))
        
        key_count += 1
        # del feat_stack_tensor

    print('STFT feature Extraction: END')

    return data_feat_cat_tensor

#########--------------------------------------------------------------------

def extract_istft():
    pass

def extract_istft_cuda():
    pass




