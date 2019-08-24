import os
import errno
import subprocess
import re
import numpy as np
import pickle
import torch
import librosa
import gc

__author__ = "Vinay Kumar"
__copyright__ = "copyright 2018, Project SSML"
__maintainer__ = "Vinay Kumar"
__status__ = "Research & Development"

#########--------------------------------------------------------------------

def replace_keys(parent_path, old_key, new_key):
    """
    Replaces all 'old_key' used in naming any file/directory with 'new_key'

    Inputs:
    - parent_path   [type=str]  : The path to parent directory where all files/directory we want to rename are stored
    - old_key       [type=str]  : old key we want to be replaced
    - new_key       [type=str]  : new key we want to replace the old key

    """

    for subdirs, dirs, files in os.walk(parent_path):
        for file in files:
            os.rename(os.path.join(subdirs, file), os.path.join(subdirs, file.replace(old_key,new_key)))
        for i in range(len(dirs)):
            new_name = dirs[i].replace(old_key,new_key)
            os.rename(os.path.join(subdirs, dirs[i]), os.path.join(subdirs, new_name))
            dirs[i] = new_name


#########--------------------------------------------------------------------

def concatenate_wav(path_master_wav_repo, path_target_wavfile, sox_script_file):
    """
    Concatenates a set of wav files using `sox` while maintaining the alphanumeric order of the paths of every wav file

    Inputs:
    - path_master_wav_repo  [type=str]  : the path of the repo where all wav files are present
    - path_target_wavfile   [type=str]  : the path of concatenated wav file we want to generate
    - sox_script_file       [type=str]  : the path of the bash script file which will contain `sox` script. MUST have `.sh` extension

    """


    # SANITY-CHECK:
    # Renaming the directories appropriately if not renamed already
    replace_keys(path_master_wav_repo, ' ', '_')
    replace_keys(path_master_wav_repo, '\'', '')    # replacing special characters
    replace_keys(path_master_wav_repo, '(', '')    # replacing special characters
    replace_keys(path_master_wav_repo, ')', '')    # replacing special characters
    replace_keys(path_master_wav_repo, '&', '_')    # replacing special characters

    sox_script = open(sox_script_file, 'w')    # ToDo: do sanity-check if the file exists or not


    for subdir, dirs, files in os.walk(path_master_wav_repo):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                sox_script.write(file_path + ' \\' + '\n')

    sox_script.close()
    subprocess.call(['sort', sox_script_file, '-o', sox_script_file])

    sox_script = open(sox_script_file, 'r')
    temp = sox_script.read()
    sox_script.close()
    sox_script = open(sox_script_file, 'w')
    sox_script.write('sox \\'+'\n')
    sox_script.write(temp)
    sox_script.write(path_target_wavfile)
    sox_script.close()


    subprocess.call(['bash', sox_script_file])

    return None

#########--------------------------------------------------------------------

def extract_duration(path, out_file):
    """
    Extracting filepath & duration for each wavfile and storing it in a text file.

    Inputs:
    - path      [type=str]  : path for 'Development' dirs of the Mixtures
    - out_file          [type=str]  : path of the meta file where the filename & duration will be stored

    Returns:
    - None

    """

    # sanity_check: check if the paths are correct
    # sanity_check: check if the out_file exists; if not then create one

    metadata_filepath_duration = open(out_file, 'w')

    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(subdir, file)
            wavfile, sampling_rate = librosa.load(file_path)
            wavfile_duration = librosa.get_duration(y=wavfile, sr=sampling_rate)
            metadata_filepath_duration.write(file_path + ' | ' + str(wavfile_duration) + '\n')

    metadata_filepath_duration.close()

    # sorting the wavfiles alphabetically to maintain order
    subprocess.call(['sort', out_file, '-o', out_file])


#########--------------------------------------------------------------------


def slice_recording(path_recording, path_metadata_filepath_duration):
    """
    Slice the wav file into chunks as-per the info present in 'path_metadata_filepath_duration`

    Inputs:
    - path_recording [type=str]: Path to the full joint recording wav file
    - path_metadata_filepath_duration [type=str]: Path to metadat storing filepath address and wav duration

    """

    metadata_filepath_duration = open(path_metadata_filepath_duration, 'r')

    start = 0.0

    for line in metadata_filepath_duration:
        filepath, duration = line.split(" | ")
        target_filepath = re.sub('/Mixtures/', '/mic_recordings/Mixtures/', filepath)
        target_parentpath = re.sub('/mixture.wav', '', target_filepath)

        # creating folder if the folder doesnot exist
        try:
            os.makedirs(target_parentpath)
        except OSERROR as exception:
            if exception.errno == errno.EEXIST and os.path.isdir(target_parentpath):
                pass

        delta_t = float(duration)

        # calling ffmpeg to slice the wav file into its respective sizes
        subprocess.call(["ffmpeg", "-i", path_recording, "-ss", str(start), "-t", str(delta_t), "-acodec", "copy", target_filepath])

        # resetting the start for next file in line
        start += delta_t

    metadata_filepath_duration.close()

#########--------------------------------------------------------------------
#########--------------------------------------------------------------------

def preprocess(dataset_name = f'MSD100',
                path_master_data_repo = '/media/sushmita/Seagate Backup Plus Drive/DataBase/musdb18hq/Mixtures/Dev/',
                path_inputs = '/media/sushmita/Seagate Backup Plus Drive/DataBase/musdb18hq/Mixtures/Dev/',
                path_labels = '/media/sushmita/Seagate Backup Plus Drive/DataBase/musdb18hq/Sources/Dev/',
                filetype_inputs = 'mixture',
                filetype_labels = 'vocals',
                stage = '',
                keys = {' ':'_', '\'':'', '(':'', ')':'', '&':'_'}
                ):
    """
    Inputs:
    - dataset_name          [type=str]  : the name of the dataset used
    - path_master_data_repo [type=str]  : the root path of the data
    - path_inputs           [type=str]  : the root path to the inputs (mixtures)
    - path_labels           [type=str]  : the root path to the labels (clean refrence sources)
    - filetype_inputs       [type=str]  : the type of input wav files (mixture, noisymix etc)
    - filetype_labels       [type=str]  : the type of label wav files (vocals, bass, drums etc)
    - stage                 [type=str]  : correct values are {dev, test, val}
    - keys                  [type=dict] : a dictionary containing the characters we want to replace as 'keys()' and new replacement characers as 'values()'

    Returns:
    - [type=dict] : a dictionary containing numpy arrays of relative paths(type=str) of inputs and labels separately
    """

    print('Data Preprocessing: START........')

    assert stage=='' or stage=='dev' or stage=='test' or stage=='val', ('ERROR: incorrect value for \'stage\' was passed. Allowed values are {dev, test, val}')

    file_inputs = f'meta_files/{dataset_name}_filenames_{stage}_{filetype_inputs}.txt'
    file_labels = f'meta_files/{dataset_name}_filenames_{stage}_{filetype_labels}.txt'

    # replacing unsupported special characters from the directory names or filenames
    for k in keys.keys():
        replace_keys(path_master_data_repo, k, keys[k])


    filenames_inputs = open(file_inputs, 'w')
    filenames_labels = open(file_labels, 'w')

    for subdir, _, files in os.walk(path_inputs):
        for file in files:
            if file.endswith(f'{filetype_inputs}.wav'):
                file_path = os.path.join(subdir, file)
                filenames_inputs.write(file_path + '\n')

    for subdir, _, files in os.walk(path_labels):
        for file in files:
            if file.endswith(f'{filetype_labels}.wav'):
                file_path = os.path.join(subdir, file)
                filenames_labels.write(file_path + '\n')

    filenames_inputs.close()
    filenames_labels.close()
    # ------------------
    subprocess.call(['sort', file_inputs, '-o', file_inputs])
    subprocess.call(['sort', file_labels, '-o', file_labels])
    # ------------------

    inputs_paths = []
    labels_paths = []
    with open(file_inputs, 'r') as file:
        for _, line in enumerate(file):
            inputs_paths.append(line.strip())

    with open(file_labels, 'r') as file:
        for _, line in enumerate(file):
            labels_paths.append(line.strip())

    inputs_paths = np.array(inputs_paths)
    labels_paths = np.array(labels_paths)

    print('Data preprocessing: END.')

    return {"inputs_paths" : inputs_paths,
            "labels_paths" : labels_paths}

#########--------------------------------------------------------------------

def postprocess(name_src, name_targets, path_src_prefix, path_src_suffix, path_output_dir, delimiter, target_tensor, key='pred_label_cat_tensor'):
    """
    Takes the frequency domain output data from the meenet model, calculates the mask and then generates the time-series data.

    Calculate:
    - masks
    - timeseries data
    - SIR(signal-to-interfernece ratio)
    - SAR(signal-to-artefacts ratio)
    - SDR(signal-to-distortion ratio)

    Inputs:
    - name_src              [type=list] :
    - name_targets          [type=list] :
    - path_src_prefix       [type=str]  :
    - path_src_suffix       [type=str]  :
    - path_output_dir
    - delimiter             [type=str]  :
    - target_tensor
    - key                   [type=str]  :

    Returns:

    """

    # generating masks
    masks = {}              # dictionary of tensor
    raw_scaled_tensor = {}
    total_tensor_sum = None
    mul_factor = 1
    add_factor = 1e-07

    i = 0
    for target in name_targets:
        for source in name_src:
            temp_out = torch.load(f'{path_src_prefix}{delimiter}{target}{delimiter}{source}{delimiter}{path_src_suffix}')[key]
            temp_out = torch.add(torch.mul(temp_out.cuda(), mul_factor), add_factor)
            raw_scaled_tensor[f'{target}{delimiter}{source}'] = temp_out.view(1025,-1)
            if i==0:
                total_tensor_sum = temp_out
                print(f'i={i}')
                i+=1
            elif i>0:
                total_tensor_sum.add(temp_out)
                print(f'i={i}')
                i+=1

    # creating masks
    for target in name_targets:
        for source in name_src:
            masks[f'{target}{delimiter}{source}'] = torch.mul(raw_scaled_tensor[f'{target}{delimiter}{source}'], mul_factor)/total_tensor_sum

        # saving the masks as pickle file
        with open(f'meta_blob/playground/{target}{delimiter}{source}.mask', 'wb') as handle:
            pickle.dump(obj=masks[f'{target}{delimiter}{source}'], file=handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'masks for {target}{delimiter}{source} is saved successfully as meta_blob/playground/{target}{delimiter}{source}.mask')

    ############################

    # generating ISTFT to get timeseries data
    # need to pass original STFT of the target tensor, NOT the absolute stft

    # for source in name_src:
    #     blueprint_shape = masks[f'{target}{delimiter}{source}'].shape
    #     pred_source_istft = librosa.istft(stft_matrix=((masks[f'{target}{delimiter}{source}']).cpu().numpy())*target_tensor.cpu().numpy()[:,0:blueprint_shape[1]],
    #                                         hop_length=512, win_length=2048, window='hann')
    #     librosa.output.write_wav(path=f'{path_output_dir}{target}{delimiter}{source}.wav', y=pred_source_istft, sr=44100)



def data_generation(csv_file, root_dir, blueprint, paths_data_tensor, filetype_inputs_list, filetype_labels_list):
    num_files = 0
    for type_input in filetype_inputs_list:
        for type_label in filetype_labels_list:
            if not torch.cuda.is_available():
                # SANITY-CHECK: whether the pre-calculated stft_features Tensors are there?
                if os.path.exists(paths_data_tensor[type_input][type_label]):
                    # Don't calculate the stft features again. Just load the pytorch tensors to the original device

                    print('features tensor file exists. Loading the extracted features tensor.')
                    # tr_data_feats_tensor = torch.load('tr_data_feats_tensor.pt', map_location='cuda:0')
                    # tr_data_feats_tensor.to(device)
                    stft_tensor = torch.load(paths_data_tensor[type_input][type_label], map_location='cpu')
                    num_files = stft_tensor.shape[1]
                    # tr_data_feats_tensor.to(device)

                    print(f'Features Tensor loaded successfully on {stft_tensor.device}.')
                    print(f'{stft_tensor.shape}')
                    for i in range(stft_tensor.shape[0]):
                        for j in range(stft_tensor.shape[1]):
                            if i==0:
                                path = f'{root_dir}mixtures/{j}.stfttensor'
                                data = stft_tensor[i,j].clone()
                                data = data.view_as(blueprint)
                                torch.save(data, path)
                                # print(i, '-@@@@', data.shape)
                            elif i==1:
                                path = f'{root_dir}{type_label}/{j}.stfttensor'
                                data = stft_tensor[i,j].clone()
                                data = data.view_as(blueprint)
                                torch.save(data, path)
                                # print(i, '-@@@@', data.shape)
                    del stft_tensor
                    gc.collect()

    # create the csv file
    obj_csv_file = open(csv_file, 'w')
    for prefix in range(num_files):
        for src in filetype_inputs_list:
            obj_csv_file.write(f'{src}/{prefix}.stfttensor')
            for lbl in filetype_labels_list:
                obj_csv_file.write(f',{lbl}/{prefix}.stfttensor')
        obj_csv_file.write(f'\n')
    obj_csv_file.close()
    print(f'Finished creating .csv file @ {csv_file}')










