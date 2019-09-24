import torch
import torch.utils.data
import os
import pandas as pd

__author__ = "Vinay Kumar"
__copyright__ = "copyright 2018, Project SSML"
__maintainer__ = "Vinay Kumar"
__status__ = "Research & Development"

class Meenet1Dataset(torch.utils.data.Dataset):
    """
    Custom dataset for meenet1 task.
    """

    def __init__(self, csv_file, root_dir, label_type, transform=None):
        """
        Inputs:
        - csv_file    [type=str] : path of the csv file which contains the information about the data
        - root_dir    [type=str] : path of the directory with all input data
        - label_type  [type=str] : allowed values {'bass', 'drums', 'others', 'vocals'}
        - transform   (callable, optional) : Transforms to be applied to the input data
        """

        self.data_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_type = label_type
        self.transform = transform

        #self.label_type_dict = {'bass':1, 'drums':2, 'other':3, 'vocals':4}
        self.label_type_dict = {'vocals':0}



    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):

        # get the input tensor
        input_data_name = os.path.join(self.root_dir, self.data_info.iloc[idx,0])
        print(input_data_name)
        input_data_tensor = torch.load(input_data_name)    # set map_location='cuda'

        # get the asked label tensor
        label_data_name = os.path.join(self.root_dir,
                                       self.data_info.iloc[idx, self.label_type_dict[self.label_type]])

        label_data_tensor = torch.load(label_data_name)    # set map_location='cuda'

        sample = {'input':input_data_tensor, 'label':label_data_tensor}

        if self.transform:
            self.transform(sample)

        return sample


