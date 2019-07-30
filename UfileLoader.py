import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


def dataloader():
    dataset_train_labels=[]
    dataset_train_data=[]
    dataset_validate_labels=[]
    dataset_validate_data=[]
    data=sio.loadmat("data\SwissRollData.mat")

    dataset_train_labels.append(data['Ct'])
    dataset_train_data.append(data['Yt'])
    dataset_validate_labels.append(data['Cv'])
    dataset_validate_data.append(data['Yv'])


    data=sio.loadmat("data\PeaksData.mat")

    dataset_train_labels.append(data['Ct'])
    dataset_train_data.append(data['Yt'])
    dataset_validate_labels.append(data['Cv'])
    dataset_validate_data.append(data['Yv'])


    data=sio.loadmat("data\GMMData.mat")

    dataset_train_labels.append(data['Ct'])
    dataset_train_data.append(data['Yt'])
    dataset_validate_labels.append(data['Cv'])
    dataset_validate_data.append(data['Yv'])

    return dataset_train_data,dataset_train_labels,dataset_validate_data,dataset_validate_labels