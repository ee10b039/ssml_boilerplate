#!usr/bin/env python
import os
import gc
import subprocess
import numpy as np
import torch
import pickle
import time
import uuid
import shutil

from modules import data_processing as dp
from modules import feature_extraction as fe
# from modules import telecast_view as tv
from modules import meenet1
from modules import helpers
from modules import dataloader as dl

__author__ = "Vinay Kumar"
__copyright__ = "copyright 2018, Project SSML"
__maintainer__ = "Vinay Kumar"
__status__ = "Research & Development"

#########--------------------------------------------------------------------------------------
######### Network TRAINING --------------------------------------------------------------------
#########--------------------------------------------------------------------------------------
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(f'Running on {device}')

# global variables
do_training = True
do_testing = False
do_batchwise_processing = True
do_cross_validation = False
generate_data = True

tr_loss_history_cuda = None

# Details of datset and network architecture
ARCH_NAME = 'MEENET1'
DATASET_NAME = 'MEENET1'
print(f'Using DATASET == {DATASET_NAME}')

audio_channel = 'mono'                      # used audio channel of wav files
filetype_inputs_list = ['mixture']
filetype_labels_list = ['vocals']

# data-generation prerequisites
#csv_file = '/scratch/sushmita.t/data/meenet1_test_data.csv'
#root_dir = '/scratch/sushmita.t/data'
stage = {'dev'}
stages = {'dev'}
csv_files = {'dev':"/scratch/sushmita.t/data/meenet1_dev_data.csv",
            }
root_dirs = {'dev':"/scratch/sushmita.t/data",
            }

paths_data_tensor = {}

# Hyperparams for network
print('Setting hyper-parameters.......')
batch_size = 100
num_workers = 100
num_epochs = 150
num_xval_folds = 5

reg_strengths = []

# sampling learning rates
learning_rates = []
for itr in range(10):
    lr = 10**np.random.uniform(-3,-6)
    learning_rates.append(lr)
print(f'------------------\nlearning_rates = {learning_rates}\n------------------')

# learning_rates = np.random.uniform(low=1e-6, high=1e-3, size=5)
print(f'learning_rates = {learning_rates}')
lr_decay_rate = 0.05        # rate at which the learning_rate decays
lr_decay_rule = 'constant_loss'
lr_decay_epoch_size = 3          # [if lr_decay_rule=True consecutive] no of epochs after which we apply lr_decay_rate

optim_hyperparams = {'adam':{'beta_1':0.9,
                            'beta_2':0.999,
                            'epsilon':1e-08,
                            'weight_decay':0.004
                            }
                    }

train_params = {'max_epochs':num_epochs,
                'batch_size':batch_size,
                'num_workers':num_workers,
                'learning_rates':learning_rates,
                'lr_decay_rate':0.05,
                'lr_decay_rule':'constant_loss',
                'reg_strengths': reg_strengths
                }

blueprint_input = torch.zeros(1,1,1025,15, device=device)

#########################################################################################

#torch.save(fe.extract_stft_cuda(device,dp.preprocess()), "meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt")
#type_input = ['mixtures']
#type_label = [ 'vocals']


#paths_data_tensor[stage][type_input][type_label] = f'/scratch/data/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt'

########################################################################################
if generate_data:
    print(f'generate_data={generate_data}')
    for stage in stages:
        paths_data_tensor[stage] = {}
        # num_files = 0
        for type_input in filetype_inputs_list:
            paths_data_tensor[stage][type_input] = {}
            for type_label in filetype_labels_list:
                if torch.cuda.is_available():
                    print(f'/scratch/data/{DATASET_NAME}_{stage}_{type_input}_{type_label}_{audio_channel}_stft_tensor.pt')
 		    # SANITY-CHECK: whether the pre-calculated stft_features Tensors are there?
                    if os.path.exists(f'/scratch/sushmita.t/data/{DATASET_NAME}_{stage}_{type_input}_{type_label}_{audio_channel}_stft_tensor.pt'):
                        print('Enter')
                        # appending the paths to be fed to data_generation module
                        paths_data_tensor[stage][type_input][type_label] = f'/scratch/sushmita.t/data/{DATASET_NAME}_{stage}_{type_input}_{type_label}_{audio_channel}_stft_tensor.pt'


        dp.data_generation(csv_file=csv_files[stage],
                            root_dir=root_dirs[stage],
                            blueprint=blueprint_input,
                            paths_data_tensor=paths_data_tensor[stage],
                            filetype_inputs_list=filetype_inputs_list,
                            filetype_labels_list=filetype_labels_list)

#########################################################################################
if do_cross_validation and do_training and do_batchwise_processing:     # neede only for dev
    # generating a unique ID for this process/job
    # stage='val'
    job_id = uuid.uuid4().hex
    dev_csv = open(csv_files['dev'])
    csv_files['xval'] = {'dev':[], 'val':[]}   # creating a placeholder for CV folds' csv paths

    num_dev_data_points = sum(1 for line in dev_csv)    # count the total number of data points in dev (lines in dev csv)
    num_data_points_per_fold = int(num_dev_data_points/num_xval_folds)      # count the number of data points for each fold

    dev_csv.close()
    dev_csv = open(csv_files['dev'])
    # generate the csv files for cross-validation
    path_csv_dir = root_dirs['dev']
    # print(dev_csv)

    anchor = 0
    fld_idx = 0

    for idx, line in enumerate(dev_csv):
        # print(f'idx={idx}')
        if fld_idx < num_xval_folds:
            # print(f'fld_idx={fld_idx}/{num_xval_folds}')
            if idx == 0:
                path_csv_fold = f'{path_csv_dir}{job_id}_val_fold{fld_idx}.csv'
                # print(f'{path_csv_fold}')
                fold_csv = open(path_csv_fold, 'w')
                csv_files['xval']['val'].append(path_csv_fold)
                csv_files['xval']['dev'].append(f'{path_csv_dir}{job_id}_dev_fold{fld_idx}.csv')
            elif idx == anchor+num_data_points_per_fold:
                fold_csv.close()
                fld_idx += 1
                anchor = idx
                path_csv_fold = f'{path_csv_dir}{job_id}_val_fold{fld_idx}.csv'
                fold_csv = open(path_csv_fold, 'w')
                csv_files['xval']['val'].append(path_csv_fold)
                csv_files['xval']['dev'].append(f'{path_csv_dir}{job_id}_dev_fold{fld_idx}.csv')

            fold_csv.write(line)    # writing each line into the fold csv file

    for i in range(len(csv_files['xval']['val'])):
        with open(csv_files['xval']['dev'][i],'wb') as wfd:
            for j, f in enumerate(csv_files['xval']['val']):
                if j != i:
                    with open(f,'rb') as fd:
                        shutil.copyfileobj(fd, wfd, 1024*1024*10)  #10MB per writing chunk to avoid reading big file into memory.
                        # print('success')
                    # fd.close()
        with open(csv_files['xval']['dev'][i], 'rb') as rfd:
            s = sum(1 for line in rfd)
            print(i, s, rfd)

    print(csv_files)
    print(f'Cross Validation folds created successfully.')
    print(f'CVfolds_csv files are saved @ "{path_csv_dir}{job_id}_val_fold*.csv"')
    print(f'dev_csv files for each val_fold are saved @ "{path_csv_dir}/{job_id}_dev_fold*.csv"')

    # create/access training-val spits
    # set the target hyper-parameter to optimize
    # run max_epochs for the every set of train-val spits
    # record the val_accuracies for each set
    # change the value of target hyper-parameter and iterate above two steps

    print(f'Running Cross Validation on {num_xval_folds} folds')
    # generate the csv file for each fold with unique id for each run & get root dir

    if torch.cuda.is_available():
        print(f'Running xCV on {device}')
        for val_fold in range(num_xval_folds):
            for type_input in filetype_inputs_list:
                for type_label in filetype_labels_list:
                    stage='dev'
                    criterion = torch.nn.MSELoss(size_average=True)
                    criterion.cuda()
                    print(f'Using MSELoss(size_average=True)')
                    print(f'stage={stage} | {type_input}/{type_label} | val_fold={val_fold}')

                    print(f'Creating DEV dataset for val_fold={val_fold}')
                    dataset = dl.Meenet1Dataset(csv_file=csv_files['xval'][stage][val_fold],
                                                root_dir=root_dirs[stage],
                                                label_type=type_label)

                    print(f'Creating the Dataloader')
                    dataloader=torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                            shuffle=True, num_workers=num_workers)

                    helpers.train_batchwise(dataloader=dataloader, arch_name = ARCH_NAME, dataset_name=DATASET_NAME,
                                            label_type=type_label, blueprint_input=blueprint_input,
                                            train_params=train_params, optim_hyperparams=optim_hyperparams,
                                            criterion=criterion,
                                            is_xCV=True, xCV_id=f'{job_id}', xCV_foldIdx=val_fold,
                                            xCV_val_csv=csv_files['xval']['val'][val_fold],
                                            xCV_root_dir=root_dirs[stage])






    else:
        print(f'CUDA device/drivers are not available. Please ensure that CUDA is available. \nStopping Cross Validation training')


if do_batchwise_processing and not do_cross_validation:
    for type_input in filetype_inputs_list:
        for type_label in filetype_labels_list:
            if torch.cuda.is_available():

                print(f'Using MEENET1_DATALOADER')
                print(f'Running on {device}')
                criterion = torch.nn.MSELoss(size_average=True)
                criterion.to(device)
                print(f'using MSELoss.')
                print(f'type_input = {type_input}, type_label = {type_label}')

                if do_training:
                    stage='dev'
                    print(f'stage = {stage} | {type_input}/{type_label}')

                    print(f'creating the {stage} dataset....HERE I AM')
                    dataset = dl.Meenet1Dataset(csv_file=csv_files[stage], root_dir=root_dirs[stage], label_type=type_label)
                    


                    print(f'creating the {stage} dataloader')
                    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                                                shuffle=True, num_workers=num_workers)
                    start = time.time()
                    helpers.train_batchwise(dataloader=dataloader, arch_name = ARCH_NAME, dataset_name=DATASET_NAME,
                                            label_type=type_label, blueprint_input=blueprint_input,
                                            train_params=train_params, optim_hyperparams=optim_hyperparams,
                                            criterion=criterion)
                    print(f'delta_time(helpers.train_batchwise[{type_label}]) = {time.time() - start}')













                # elif do_testing:
                #     stage='test'
                #     print(f'stage={stage} | {type_input}/{type_label}')

                #     print(f'creating the {stage} dataset')
                #     dataset = dl.Meenet1Dataset(csv_file=test_csv_file, root_dir=test_root_dir, label_type=type_label)

                #     start = time.time()
                #     helpers.test_batchwise(model=)

    ##################### OLDER-CODE ###################################################################

else:
    for type_input in filetype_inputs_list:
        for type_label in filetype_labels_list:
            if torch.cuda.is_available():
                # run CUDA scripts
                print(f'Running on {device}.')
                print(f'Setting up criterion.......')
                # Mean Square Loss (without division by 'n')
                criterion = torch.nn.MSELoss(size_average=False)
                criterion.to(device)
                print(f'Using MSELoss as evaluation startegy')

                print(f'type_input = {type_input}, type_label = {type_label}')

                if do_training:
                    stage='dev'
                    print(f'stage = {stage} | {type_input}/{type_label} | Step-0: PREPROCESS ')
                    dev_data_paths = dp.preprocess(dataset_name=DATASET_NAME,
                                                    path_master_data_repo='../data/MSD100/',
                                                    path_inputs='../data/MSD100/Mixtures/Dev/',
                                                    path_labels='../data/MSD100/Sources/Dev/',
                                                    filetype_inputs=type_input,
                                                    filetype_labels=type_label,
                                                    stage=stage
                                                    )

                    # SANITY-CHECK: whether the pre-calculated stft_features Tensors are there?
                    if os.path.exists(f'meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt'):
                        # Don't calculate the stft features again. Just load the pytorch tensors to the original device
                        print('features tensor file exists. Loading the extracted features tensor.')
                        # tr_data_feats_tensor = torch.load('tr_data_feats_tensor.pt', map_location='cuda:0')
                        # tr_data_feats_tensor.to(device)
                        dev_stft_tensor = torch.load(f'meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt', map_location='cpu')
                        # tr_data_feats_tensor.to(device)
                        print(f'Features Tensor loaded successfully on {dev_stft_tensor.device}.')
                    else:
                        print('features tensor file DOESNOT exists. Initiating the Feature Extraction process.....')
                        print('Step-1:')

                        if audio_channel == 'mono':
                            dev_stft_tensor = fe.extract_stft_cuda(device=device, data_paths=dev_data_paths, to_mono=True)
                        elif audio_channel == 'stereo':
                            dev_stft_tensor = fe.extract_stft_cuda(device=device, data_paths=dev_data_paths, to_mono=False)

                        # saving the STFT feature tensor as pytorch file
                        torch.save(dev_stft_tensor, f'meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt')
                        print(f'Extracted features saved succesfully at: meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt')

                    print('Step-2:')
                    dev_loss_history_cuda = helpers.train_cuda(device=device, dataset_name=DATASET_NAME, audio_channel=audio_channel,
                                                                type_input=type_input, type_label=type_label,
                                                                data_feats_tensor=dev_stft_tensor, blueprint_input=blueprint_input,
                                                                criterion=criterion, num_epochs=num_epochs, learning_rates=learning_rates,
                                                                lr_decay_rate = lr_decay_rate, lr_decay_epoch_size=lr_decay_epoch_size,
                                                                optim_hyperparams=optim_hyperparams , reg_strengths=reg_strengths,
                                                                batch_size=batch_size)

                    del dev_stft_tensor
                    gc.collect()

                    # telecast the loss v/s epoch for each learning rate
                    # tv.static_screen(phase='train', data=tr_loss_history_cuda, save_plot=False)

                ######### Network TESTING @ CUDA -------------------------------------------------------------

                if do_testing:
                    stage = 'test'
                    m_lr_test = 2e-05
                    print(f'stage = {stage} | {type_input}/{type_label} | TESTING.... ')
                    test_data_paths = dp.preprocess(dataset_name=DATASET_NAME,
                                                    path_master_data_repo='../data/MSD100/',
                                                    path_inputs='../data/MSD100/Mixtures/Test/',
                                                    path_labels='../data/MSD100/Sources/Test/',
                                                    filetype_inputs=type_input,
                                                    filetype_labels=type_label,
                                                    stage=stage
                                                    )

                    # SANITY-CHECK: whether the pre-calculated stft_features Tensors are there?
                    if os.path.exists(f'meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt'):
                        # Don't calculate the stft features again. Just load the pytorch tensors to the original device
                        print('features tensor file exists. Loading the extracted features tensor')
                        test_stft_tensor = torch.load(f'meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt')
                        print('Features Tensor loaded successfully.')
                    else:
                        print('features tensor file DOESNOT exists. Initiating the Feature Extraction process.....')
                        print('Step-1:')

                        if audio_channel == 'mono':
                            test_stft_tensor = fe.extract_stft_cuda(device=device, data_paths=test_data_paths, to_mono=True)
                        elif audio_channel == 'stereo':
                            test_stft_tensor = fe.extract_stft_cuda(device=device, data_paths=test_data_paths, to_mono=False)

                        # saving to a pytorch file
                        torch.save(test_stft_tensor, f'meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt')
                        print(f'Extracted features saved succesfully at: meta_blob/{DATASET_NAME}_{stage}_{type_input}-{type_label}_{audio_channel}_stft_tensor.pt')

                    print('Step-2: Starting TESTING process')
                    test_output_cuda = helpers.test_cuda(device=device,
                                                        path_model=f'trained_models/meenet1_{DATASET_NAME}_{type_input}-{type_label}_{audio_channel}_{m_lr_test}_{num_epochs}epochs_{device}.model',
                                                        test_feats_tensor=test_stft_tensor, criterion=criterion,
                                                        blueprint_input=blueprint_input)

                    del test_stft_tensor
                    gc.collect()

                    # saving the test outputs (tensor) in a file
                    print('Saving the Test outputs in a file')
                    torch.save(test_output_cuda, f'meta_blob/meenet1_{DATASET_NAME}_output_{stage}_{type_input}_{type_label}_{audio_channel}_{m_lr_test}_{device}.pt')
                    print(f'Test outputs saved successfully at: meta_blob/meenet1_{DATASET_NAME}_output_{stage}_{type_input}_{type_label}_{audio_channel}_{m_lr_test}_{device}.pt')
                    print('TESTING FINISHED.')

                #########--------------------------------------------------------------------------------------
                #########--------------------------------------------------------------------------------------
                #########--------------------------------------------------------------------------------------
                #########--------------------------------------------------------------------------------------

            else:
                # run CPU scripts
                print(f'NO GPU devices available. ABORTING Training & Testing')




