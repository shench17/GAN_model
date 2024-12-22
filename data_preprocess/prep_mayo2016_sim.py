import os
import argparse
import numpy as np
import h5py
import pandas as pd
import pydicom
from scipy.io import savemat
import cv2

def save_dataset(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))
    df = pd.read_csv(os.path.join(args.data_path,'train_data2.csv'))
    patients_list = df['train'].tolist()
    #patients_list = sorted([d for d in os.listdir(args.data_path) if df in d])
    for p_ind, patient in enumerate(patients_list):
        #print(patient)
        if p_ind >= 0:
            #patient_id = patient.split('_', 1)[0]
            patient_path = os.path.join(args.data_path+"/Coltea-Lung-CT-100W/", patient+"/NATIVE/DICOM")
            '''Img = h5py.File(patient_path)
            print(Img)
            Img = Img['Img_CT'][:]   # (slices ,512, 512)

            io = 'target'
            '''
            for slice in os.listdir(patient_path):
                f_name = '{}_{}_{}.npy'.format('NATIVE',patient,slice)
                ds = pydicom.dcmread(os.path.join(patient_path,slice))
                #print(ds)
                # 获取DICOM文件的图像数据
                pixel_array = ds.pixel_array
                
                #pixel_array= cv2.resize(pixel_array, (256, 256))
                pixel_array[pixel_array < 0] = 0
                pixel_array = pixel_array / 1e3
                pixel_array = pixel_array - 1
                np.save(os.path.join(args.save_path, f_name), pixel_array.astype(np.uint16))

    for p_ind, patient in enumerate(patients_list):
        #print(patient)
        if p_ind >= 0:
            #patient_id = patient.split('_', 1)[0]
            patient_path = os.path.join(args.data_path+"/Coltea-Lung-CT-100W/", patient+"/ARTERIAL/DICOM")

            for slice in os.listdir(patient_path):
                f_name = '{}_{}_{}.npy'.format('ARTERIAL',patient,slice)
                ds = pydicom.dcmread(os.path.join(patient_path,slice))
                #print(ds)
                # 获取DICOM文件的图像数据
                pixel_array = ds.pixel_array
                #pixel_array= cv2.resize(pixel_array, (256, 256))
                pixel_array[pixel_array < 0] = 0
                pixel_array = pixel_array / 1e3
                pixel_array = pixel_array - 1
                np.save(os.path.join(args.save_path, f_name), pixel_array.astype(np.uint16))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='D:/database/Coltea-Lung-CT-100W/Coltea-Lung-CT-100W')   # data format: matlab
    parser.add_argument('--save_path', type=str, default='./gen_data/CT/')
    parser.add_argument('--dose', type=int, default=5)
    args = parser.parse_args()

    save_dataset(args)
