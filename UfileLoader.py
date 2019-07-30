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
    plt.scatter(data['Yt'][0,:],data['Yt'][1,:],c=np.argmax(data['Ct'],axis=0))
    plt.show()

    data=sio.loadmat("data\PeaksData.mat")

    dataset_train_labels.append(data['Ct'])
    dataset_train_data.append(data['Yt'])
    dataset_validate_labels.append(data['Cv'])
    dataset_validate_data.append(data['Yv'])
    plt.scatter(data['Yt'][0,:],data['Yt'][1,:],c=np.argmax(data['Ct'],axis=0))
    plt.show()

    data=sio.loadmat("data\GMMData.mat")

    dataset_train_labels.append(data['Ct'])
    dataset_train_data.append(data['Yt'])
    dataset_validate_labels.append(data['Cv'])
    dataset_validate_data.append(data['Yv'])
    plt.scatter(data['Yt'][0,:],data['Yt'][1,:],c=np.argmax(data['Ct'],axis=0))
    plt.show()
    return dataset_train_data,dataset_train_labels,dataset_validate_data,dataset_validate_labels