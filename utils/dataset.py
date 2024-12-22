import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
import pandas as pd

class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True):
        self.mode = mode
        self.context = context
        #print(dataset)

        if dataset == 'mayo_2016':
            data_root = 'C:/Users/shench17/Desktop/CoreDiff-main/data_preprocess/gen_data/CT/'

            patient_ids = []
            if mode == 'train':
                #patient_ids.pop(test_id)
                df = pd.read_csv("D:/database/Coltea-Lung-CT-100W/Coltea-Lung-CT-100W/train_data2.csv")
                patient_ids = df['train'].tolist()

            elif mode == 'test':
                df = pd.read_csv('D:/database/Coltea-Lung-CT-100W/Coltea-Lung-CT-100W/test_data_ori.csv')
                patient_ids = df['test'].tolist()

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('NATIVE_{}_'.format(id) + '*.npy'))))
                patient_lists = patient_lists + patient_list[0:len(patient_list) - 1]
                #if(ind==1):
                    #print(patient_lists)
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('ARTERIAL_{}_'.format(id) + '*.npy'))))
                patient_list = patient_list[0:len(patient_list) - 1]
                patient_lists = patient_lists + patient_list
            base_input = patient_lists

        elif dataset == 'mayo_2020':
            data_root = './data_preprocess/gen_data/mayo_2020_npy'
            if dose == 10:
                patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050']
            elif dose == 25:
                patient_ids = ['L077', 'L056', 'L186', 'L006', 'L148']

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, (id + '_target_' + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[0:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, (id + '_{}_'.format(dose) + '*_img.npy'))))
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
                base_input = patient_lists


        elif dataset == 'piglet':
            data_root = './data_preprocess/gen_data/piglet_npy'

            patient_list = sorted(glob(osp.join(data_root, 'piglet_target_' + '*_img.npy')))
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(data_root, 'piglet_{}_'.format(dose) + '*_img.npy')))
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                    base_input = cat_patient_list
            else:
                patient_list = patient_list[1:len(patient_list) - 1]
                base_input = patient_list


        elif dataset == 'phantom':
            data_root = './data_preprocess/gen_data/xnat_npy'

            patient_list = sorted(glob(osp.join(data_root, 'xnat_target' + '*_img.npy')))[9:21]
            base_target = patient_list[1:len(patient_list) - 1]

            patient_list = sorted(glob(osp.join(data_root, 'xnat_{:0>3d}_'.format(dose) + '*_img.npy')))[9:21]
            if context:
                cat_patient_list = []
                for i in range(1, len(patient_list) - 1):
                    patient_path = ''
                    for j in range(-1, 2):
                        patient_path = patient_path + '~' + patient_list[i + j]
                    cat_patient_list.append(patient_path)
                    base_input = cat_patient_list
            else:
                patient_list = patient_list[1:len(patient_list) - 1]
                base_input = patient_list

        self.input = base_input
        self.target = base_target
        #print(len(self.input))
        #print(len(self.target))


    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]

        
        input = np.load(input)[np.newaxis, ...].astype(np.float32) #(1, 512, 512)
        target = np.load(target)[np.newaxis,...].astype(np.float32) #(1, 512, 512)

        input = self.normalize_(input)
        target = self.normalize_(target)

        return input, target

    def __len__(self):
        return len(self.target)

    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
        img = img - 1024
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        img = (img - MIN_B) / (MAX_B - MIN_B)
        return img


dataset_dict = {
    'train': partial(CTDataset, dataset='mayo_2016', mode='train', test_id=9, dose=5, context=True),
    'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=None, dose=5, context=True),
    'mayo_2020': partial(CTDataset, dataset='mayo_2020', mode='test', test_id=None, dose=None, context=True),
    'piglet': partial(CTDataset, dataset='piglet', mode='test', test_id=None, dose=None, context=True),
    'phantom': partial(CTDataset, dataset='phantom', mode='test', test_id=None, dose=108, context=True),
}
