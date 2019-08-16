import numpy as np
import torch
import pickle
import gc
from modules import meenet1
from modules import dataloader as dl

__author__ = "Vinay Kumar"
__copyright__ = "copyright 2018, Project SSML"
__maintainer__ = "Vinay Kumar"
__status__ = "Research & Development"

#########--------------------------------------------------------------------

# weight initializations
def weights_init(model):
    """
    Initializes the weights and bias of the the network.
    """
    if isinstance(model, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(model.weight)
        print(f'weights initialized as per "Xavier Initialization Rule"')


def test_batchwise(path_model, dataloader, label_type, blueprint_input, criterion, arch_name, dataset_name, is_xCV=False, xCV_id=None):
    """
    Batchwise testing and validation

    Inputs:
    - path_model        [type=torch.nn.Module]
    - dataloader        [type=torch.utils.data.DataLoader]
    - label_type        [type=str]
    - blueprint_input   [type=torch.Tensor]
    - criterion         [type=torch.nn.*]
    - arch_name         [type=str]
    - dataset_name      [type=str]
    - is_xCV            [type=bool]
    - xCV_id            [type=str]

    Returns:
    - loss              [type=dict]     : the loss obtained over the test or validation data
    """

    if torch.cuda.is_available():
        if arch_name == 'MEENET1':
            print('LOADING model for testing......')
            model_meenet1 = meenet1.MeeAutoEncoder()
            model_meenet1.cuda()
            # model_meenet1.load_state_dict(torch.load(path_model)) # used when we save a model's state_dict()
        model = torch.load(path_model)      # used when we have saved a whole model using model.save()

        num_data_points = 0
        running_loss = 0
        # loss_history = []
        for _, sample_batched in enumerate(dataloader):
            # print(f'i_batch={i_batch}')
            # print(sample_batched['input'].size())
            # print(sample_batched['label'].size())

            input_batch = sample_batched['input'].cuda().view(-1,1,1025,15)
            label_batch = sample_batched['label'].cuda().view(-1,1,1025,15)
            # print(input_batch.device)
            # print(input_batch.shape)

            num_data_points += input_batch.shape[0]

            pred_label = model(input_batch)
            # print(f'pred_label.shape = {pred_label.shape}')

            loss = criterion(pred_label, label_batch)
            # print(loss.item())
            running_loss += loss.item()

        return running_loss/num_data_points
    else:
        print(f'CUDA not available. Stopping execution.')
        print(f'Please ensure that CUDA device/drivers are available.')
        # exit()



def train_batchwise(dataloader, arch_name, dataset_name, label_type, blueprint_input, train_params, optim_hyperparams, criterion, is_xCV=False, xCV_id=None, xCV_val_csv=None, xCV_foldIdx=None, xCV_root_dir=None):
    """
    Batchwise training of the network.
    """

    max_epochs = train_params['max_epochs']
    batch_size = train_params['batch_size']
    label_type = label_type

    print('Network Initialization.......')
    model_meenet1 = meenet1.MeeAutoEncoder()
    model_meenet1.cuda()

    # counting the number of trainable & total parameters
    meenet1_num_total_params = sum(p.numel() for p in model_meenet1.parameters())
    meenet1_num_trainable_params = sum(p.numel() for p in model_meenet1.parameters() if p.requires_grad)
    print(f'Number of TOTAL parameters in meenet1 = {meenet1_num_total_params}')
    print(f'Number of TRAINABLE parameters in meenet1 = {meenet1_num_trainable_params}')

    loss_history = {}
    if is_xCV:
        val_loss_history = {}

    for lr in train_params['learning_rates']:
        print(f'lr={lr}')

        model_meenet1.apply(weights_init)
        loss_history[lr] = []

        optimizer = torch.optim.Adam(params=model_meenet1.parameters(),
                                    lr=lr,
                                    betas=(optim_hyperparams['adam']['beta_1'], optim_hyperparams['adam']['beta_2']),
                                    eps=optim_hyperparams['adam']['epsilon'],
                                    weight_decay=optim_hyperparams['adam']['weight_decay'],
                                    amsgrad=True
                                    )
        max_epochs = train_params['max_epochs']

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1.1)
        scheduler_name = 'none'

        for epoch in range(max_epochs):
            running_loss = 0.0
            num_train = 0

            # sampling the batch
            for _, sample_batched in enumerate(dataloader):
                # print(f'i_batch={i_batch}')
                # print(sample_batched['input'].size())
                # print(sample_batched['label'].size())

                input_batch = sample_batched['input'].cuda().view(-1,1,1025,15)
                label_batch = sample_batched['label'].cuda().view(-1,1,1025,15)
                # print(input_batch.device)
                # print(input_batch.shape)

                num_train += input_batch.shape[0]

                optimizer.zero_grad()

                pred_label = model_meenet1(input_batch)
                # print(f'pred_label.shape = {pred_label.shape}')

                loss = criterion(pred_label, label_batch)
                # print(loss.item())
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            # if epoch >= 50:
            # scheduler.step()

            loss_history[lr].append(running_loss/num_train)
            print(f'lr={lr} : [epoch {epoch}/{max_epochs}] : loss = {loss_history[lr][-1]}')

            # print('lr = {} : [epoch {}/{}] : loss = {}'.format(lr, epoch, num_epochs, loss_history[m_lr][-1]))

        if is_xCV:
            path_trained_model = f'trained_models/{arch_name}_{dataset_name}_{label_type}_{scheduler_name}_{lr}lr_{max_epochs}epochs_{batch_size}batches_{xCV_id}_dev_fold{xCV_foldIdx}.model'
        else:
            path_trained_model = f'trained_models/{arch_name}_{dataset_name}_{label_type}_{scheduler_name}_{lr}lr_{max_epochs}epochs_{batch_size}batches.model'

        torch.save(model_meenet1, path_trained_model)
        print(f'Trained model is saved @ {path_trained_model}')

        if is_xCV:
            # do cross validation test and report the loss
            print(f'Creating VAL dataset for val_fold={xCV_foldIdx}')
            val_dataset = dl.Meenet1Dataset(csv_file=xCV_val_csv,
                                        root_dir=xCV_root_dir,
                                        label_type=label_type)

            print(f'Creating the Dataloader fo VAL datatset')
            val_dataloader=torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=train_params['num_workers'])

            val_loss = test_batchwise(path_model=path_trained_model, dataloader=val_dataloader,
                            label_type=label_type, blueprint_input=blueprint_input, criterion=criterion,
                            arch_name=arch_name, dataset_name=dataset_name,
                            is_xCV=is_xCV, xCV_id=xCV_id)

            val_loss_history[lr] = val_loss

        # path_loss_history = f'meta_blob/{arch_name}_{dataset_name}_{label_type}_{scheduler_name}_{lr}lr_{max_epochs}epochs_{batch_size}batches.losshistory'
        # with open(path_loss_history, 'wb') as handle:
        #     pickle.dump(obj=loss_history, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f'Training Loss History is saved @ {path_loss_history}')
    if is_xCV:
        path_loss_history = f'meta_blob/{arch_name}_{dataset_name}_{label_type}_{scheduler_name}_{max_epochs}epochs_{batch_size}batches_{xCV_id}_dev_fold{xCV_foldIdx}.losshistory'
        path_val_loss_history = f'meta_blob/{arch_name}_{dataset_name}_{label_type}_{scheduler_name}_{max_epochs}epochs_{batch_size}batches_{xCV_id}_val_fold{xCV_foldIdx}.losshistory'

        with open(path_loss_history, 'wb') as handle:
            pickle.dump(obj=loss_history, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Training Loss History for xCV_fold-{xCV_foldIdx} is saved @ {path_loss_history}')

        with open(path_val_loss_history, 'wb') as handle:
            pickle.dump(obj=val_loss_history, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Validation Loss History for fold-{xCV_foldIdx} is saved @ {path_val_loss_history}')
    else:
        path_loss_history = f'meta_blob/{arch_name}_{dataset_name}_{label_type}_{scheduler_name}_{max_epochs}epochs_{batch_size}batches.losshistory'

        with open(path_loss_history, 'wb') as handle:
            pickle.dump(obj=loss_history, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Training Loss History is saved @ {path_loss_history}')



# training routine on CUDA device
def train_cuda(device, dataset_name, audio_channel, type_input, type_label, data_feats_tensor, blueprint_input, criterion, num_epochs, learning_rates, lr_decay_rate, lr_decay_epoch_size, optim_hyperparams, reg_strengths=0.0, batch_size=1):
    """
    Trains the network.

    Inputs:
    - device            [type=torch.device] : The device we want to run the network on
    - dataset_name      [type=str]          : name of the dataset (eg. 'MSD100', 'DSD100' etc)
    - audio_channel     [type=str]          : valid values {'mono', 'stereo'}
    - type_input        [type=str]          :
    - type_label        [type=str]          :
    - data_feats_tensor [type=torch.Tensor] : The input(+labels) you want the network to train on
    - blueprint_input   [type=torch.Tensor] : Blueprint as to how we want to feed the input to network
    - criterion         [type=torch._Loss]  : loss criterion eg. MSELoss or L1 or L2 etc....
    - num_epochs        [type=int]          : total no. of epochs
    - learning_rates    [type=array]        : an array of learning rates
    - lr_decay_rate     [type=float]        :
    - lr_decay_epoch_size [type=int]        :
    - optim_hyperparams [type=dict]         : hyperparams for the optimizer
    - reg_strengths     [type=float64]      : regularization strengths
    - batch_size        [type=int]          : batch size

    Returns:
    - loss_history [type=dict] : Returns loss history for each epoch and each learning rate
    """

    print('Network Initialization.......')
    model_meenet1 = meenet1.MeeAutoEncoder()
    model_meenet1.cuda()

    # counting the number of trainable & total parameters
    meenet1_num_total_params = sum(p.numel() for p in model_meenet1.parameters())
    meenet1_num_trainable_params = sum(p.numel() for p in model_meenet1.parameters() if p.requires_grad)
    print(f'Number of TOTAL parameters in meenet1 = {meenet1_num_total_params}')
    print(f'Number of TRAINABLE parameters in meenet1 = {meenet1_num_trainable_params}')

    loss_history = {}
    num_train = data_feats_tensor.shape[1]


    for m_lr in learning_rates:

        # Initializing network weights and bias
        model_meenet1.apply(weights_init)
        loss_history[m_lr] = []

        optimizer = torch.optim.Adam(params=model_meenet1.parameters(),
                                        lr=m_lr,
                                        betas=(optim_hyperparams['adam']['beta_1'], optim_hyperparams['adam']['beta_2']),
                                        eps=optim_hyperparams['adam']['epsilon'],
                                        weight_decay=optim_hyperparams['adam']['weight_decay'],
                                        amsgrad=True
                                    )

        # optimizer = torch.optim.SGD(params=model_meenet1.parameters(),
        #                             lr=m_lr, momentum=0.75, dampening=0,
        #                             nesterov=True
        #                            )

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5,10,15,20,25,30], gamma=0.05)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1.05)
        for epoch in range(num_epochs):

            # TODO: use lr_decay_rate here
            # if len(loss_history[m_lr]) > lr_decay_epoch_size:
            #     for i in range(lr_decay_epoch_size):
            #         if loss_history[m_lr][-1*(i+1)] == loss_history[m_lr][-1*(i+1)-1]:
            #             print(f'\tloss is saturating')
            #             # i += 1
            #         else:
            #             print(f'\tNO saturation. Continuing.')
            #             break

            #         if i == lr_decay_epoch_size-1:
            #             print(f'\tloss is saturated for {lr_decay_epoch_size} epochs.\n\tDecaying lr by {lr_decay_rate}')
            #             m_lr *= lr_decay_rate
            #             print(f'\tNew learning_rate = {m_lr}')
            #             loss_history[m_lr] = []

            ####### decaying learning rate every few epochs
            # if epoch%25 == 0 and epoch != 0:
            #     print(f'decaying lr by {lr_decay_rate}')
            #     m_lr *= lr_decay_rate
            #     print(f'New lr is {m_lr}')
            #     loss_history[m_lr] = []


            scheduler.step()

            running_loss = 0.0
            for i in range(num_train):
                train_input_tensor = data_feats_tensor[0,i,:,:].cuda().view_as(blueprint_input)    # torch.Size([1, 1, 1025, 15])
                # train_input_tensor = train_input_tensor.cuda()                              # converting to 'type = torch.cuda.FloatTensor'
                train_label_tensor = data_feats_tensor[1,i,:,:].cuda().view_as(blueprint_input)    # torch.Size([1, 1, 1025, 15])
                # train_label_tensor = train_label_tensor.cuda()                              # converting to 'type = torch.cuda.FloatTensor'

                optimizer.zero_grad()

                pred_label_tensor = model_meenet1(train_input_tensor)   # torch.Size([1, 1, 1025, 15])
                # pred_label_tensor.cuda()
                # print(f'pred_label_tensor.device = {pred_label_tensor.device}')

                loss = criterion(pred_label_tensor, train_label_tensor)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            loss_history[m_lr].append(running_loss/num_train)
            print('lr = {} : [epoch {}/{}] : loss = {}'.format(m_lr, epoch, num_epochs, loss_history[m_lr][-1]))


        print('Network Training: FINISHED')
        print(f'SAVING Trained Model')
        torch.save(model_meenet1.state_dict(), f'trained_models/leakyReLU_expLR_meenet1_{dataset_name}_{type_input}-{type_label}_{audio_channel}_{m_lr}_{num_epochs}epochs_{device}.model')
        print(f'Trained Model SAVED at: trained_models/leakyReLU_expLR_meenet1_{dataset_name}_{type_input}-{type_label}_{audio_channel}_{m_lr}_{num_epochs}epochs_{device}.model')

    print(f'Saving the loss_history as pickle file....')
    with open(f'meta_blob/loss_history_leakyReLU_expLR_meenet1_{dataset_name}_{type_input}-{type_label}_{audio_channel}_{num_epochs}epochs_{device}.pkl', 'wb') as handle:
        pickle.dump(obj=loss_history, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'loss_history SAVED at: meta_blob/loss_history_leakyReLU_expLR_meenet1_{dataset_name}_{type_input}-{type_label}_{audio_channel}_{num_epochs}epochs_{device}.pkl')

    return loss_history

#########--------------------------------------------------------------------

def test_cuda(device, path_model, test_feats_tensor, criterion, blueprint_input):
    """
    Load the model and perform testing for given test inputs

    Inputs
    - device                [type=torch.device]     : CPU or CUDA
    - path_model            [type=str]              : The path of the model where model is saved
    - test_feats_tensor     [type=torch.Tensor]     : A dictionary containing the stft features in stacked format for every data
    - criterion             [type=torch._Loss]      : loss evaluating criterion eg. MSELoss etc....
    - blueprint_input       [type=torch.Tensor]     : The dimension of Tensor used by model

    Returns:
    - [type=dict] : loss for every test data and the predicted labels (torch.tensor)

    """
    print('LOADING model for testing......')
    model_meenet1 = meenet1.MeeAutoEncoder()
    model_meenet1.to(device)
    # model_meenet1.load_state_dict(torch.load(path_model)) # used when we save a model's state_dict()

    model_meenet1 = torch.load(path_model)      # used when we have saved a whole model using model.save()

    print('Trained model successfully loaded for Testing.')

    pred_label_cat_tensor = None
    loss_history = []
    num_test = test_feats_tensor.shape[1]
    segment_size = int(num_test*0.05)        # 5% of the total sample size

    Y_list =[]
    temp_cat = None
    j = 0

    with torch.no_grad():
        for i in range(num_test):
            test_input_tensor = test_feats_tensor[0,i,:,:].cuda().view_as(blueprint_input)    # torch.Size([1, 1, 1025, 15])
            test_label_tensor = test_feats_tensor[1,i,:,:].cuda().view_as(blueprint_input)    # torch.Size([1, 1, 1025, 15])
            # print(f'test_input_tensor: {test_input_tensor.device}, {type(test_input_tensor)}')
            # print(f'test_label_tensor: {test_label_tensor.device}, {type(test_label_tensor)}')

            # predicted masks (used for post-processing)
            pred_label_tensor = model_meenet1(test_input_tensor)          # has same type and shape as 'blueprint_input'
            # pred_label.to(device)
            # print(f'pred_label_tensor.device = {pred_label_tensor.device}, {type(pred_label_tensor)}')

            loss = criterion(pred_label_tensor, test_label_tensor)
            # print(f'Testvector(i)={i}: loss = {loss.item()}')

            loss_history.append(loss.item())
            # pred_label_tensor = pred_label_tensor.to(torch.device("cpu"))         # moving the predicted label tensor to CPU
            # print(f'pred_label_tensor.device = {pred_label_tensor.device}, {type(pred_label_tensor)}')

            if j<segment_size or i<=num_test-1:
                if j==0:
                    temp_cat = pred_label_tensor
                    j+=1
                elif j==segment_size-1 or i==num_test-1:
                    print(f'i={i}, j={j} | i==num_test-1 : {i==num_test-1}; j==segment_size-1 : {j==segment_size-1}')
                    temp_cat = torch.cat((temp_cat, pred_label_tensor), dim=1)
                    Y_list.append(temp_cat)
                    print(f'temp_cat.shape = {temp_cat.shape}, len(Y_list) = {len(Y_list)}')
                    temp_cat=None
                    j=0
                    gc.collect()
                else:
                    temp_cat = torch.cat((temp_cat, pred_label_tensor), dim=1)
                    j+=1

        pred_label_cat_tensor = Y_list.pop(0).cpu()
        while len(Y_list)>0:
            print(f'len(Y_list) = {len(Y_list)}')
            pred_label_cat_tensor = torch.cat((pred_label_cat_tensor, Y_list.pop(0).cpu()), dim=1 )

    print('\nTest set: num_test = {} , Average loss = {:.4f}\n'.format(num_test, np.mean(loss_history)))
    print('TESTING FINISHED. Returning loss_history.')

    return {'loss_history' : loss_history,
            'pred_label_cat_tensor' : pred_label_cat_tensor
            }

######### --------------------------------------------------------------------
######### --------------------------------------------------------------------
######### --------------------------------------------------------------------


def train(data_feats, data_keys, blueprint_input, criterion, num_epochs, num_train, learning_rates, optim_hyperparams, reg_strengths=0.0, batch_size=1):
    """
    Trains the network.

    Inputs:
    - data_feats        [type=dict]         : The input(+labels) you want the network to train on
    - data_keys         [type=array         : The keys of the data_feats dictionary
    - blueprint_input   [type=torch.Tensor] : Blueprint as to how we want to feed the input to network
    - criterion         [type=torch._Loss]  : loss criterion eg. MSELoss or L1 or L2 etc....
    - num_epochs        [type=int]          : total no. of epochs
    - num_train         [type=int]          : total nuber of training slices/data
    - learning_rates    [type=array]        : an array of learning rates
    - optim_hyperparams [type=dict]         : hyperparams for the optimizer eg. momentum, weightdecay for nesterov momentum update
    - reg_strengths     [type=float64]      : regularization strengths
    - batch_size        [type=int]          : batch size

    Returns:
    - loss_history [type=dict] : Returns loss history for each eopch and each learning rate
    """

    print('Network Initialization.......')
    model_meenet1 = meenet1.MeeAutoEncoder()

    loss_history = {}

    for m_lr in learning_rates:
        loss_history[m_lr] = []
        optimizer = torch.optim.SGD(model_meenet1.parameters(), lr=m_lr, nesterov=True, momentum=optim_hyperparams['momentum'],
                                    weight_decay=optim_hyperparams['weight_decay'], dampening=optim_hyperparams['dampening'])
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(num_train):
                train_input = torch.from_numpy(data_feats[data_keys[0]][:,:,i])
                train_input = train_input.view_as(blueprint_input)

                train_label = torch.from_numpy(data_feats[data_keys[1]][:,:,i])
                train_label = train_label.view_as(blueprint_input)
                optimizer.zero_grad()

                pred_label = model_meenet1(train_input)

                loss = criterion(pred_label, train_label)
                loss.backward()
                optimizer.step()

                running_loss += loss

            loss_history[m_lr].append(running_loss/num_train)
            print('lr = {} : [epoch {}/{}] : loss = {}'.format(m_lr, epoch, num_epochs, loss_history[m_lr][epoch]))

    print('Network Training: FINISHED', '\n SAVING Trained Model as \'meenet1_state1.model\' ....')
    torch.save(model_meenet1.state_dict(), 'meenet1_state1.model')
    print('Trained Model SAVED.....')

    print(f'Saving the loss_history as pickle file....')
    with open('train_loss_history.pkl', 'wb') as handle:
        pickle.dump(obj=loss_history, file=handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'loss_history SAVED as train_loss_history.pkl file')


    return loss_history


#########--------------------------------------------------------------------

def test(path_model, test_feats, test_keys, num_test, criterion, blueprint_input):
    """
    Load the model and perform testing for given test inputs

    Inputs
    - path_model    [type=str]  : The path of the model where model is saved
    - test_feats    [type=dict] : A dictionary containing the stft features in stacked format for every data
    - test_keys     [type=array]: Array containing the keys for test_feats dict
    - num_test      [type=int]  : total number of (1025x15) arrays in the test_feats dictionary
    - criterion     [type=torch._Loss]: loss evaluating criterion eg. MSELoss etc....
    - blueprint_input [type=torch.Tensor]: The dimension of Tensor used by model

    Returns:
    - [type=array] : loss for every test data

    """
    print('LOADING model for testing......')
    model_meenet1 = meenet1.MeeAutoEncoder()
    model_meenet1.load_state_dict(torch.load(path_model))

    loss_history = []

    with torch.no_grad():
        for i in range(num_test):
            test_input = torch.from_numpy(test_feats[test_keys[0]][:,:,i])
            test_input = test_input.view_as(blueprint_input)

            test_label = torch.from_numpy(test_feats[test_keys[1]][:,:,i])
            test_label = test_label.view_as(blueprint_input)

            # predicted masks (used for post-processing)
            pred_label = model_meenet1(test_input)          # has same type and shape as 'blueprint_input'

            loss = criterion(pred_label, test_label)
            loss_history.append(loss)

    print('\nTest set: num_test = {} , Average loss = {:.4f}\n'.format(num_test, np.mean(loss_history)))
    print('TESTING FINISHED. Returning loss_history.')

    return loss_history

######### --------------------------------------------------------------------
