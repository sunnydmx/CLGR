import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from sktime.datasets import load_from_tsfile
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t", dtype='float32')
    return data

def read(dataset_name='ECG5000'):
    filenameTSV1=r'./UCRArchive_2018/'+dataset_name+'/'+dataset_name+r'_TRAIN.tsv'
    filenameTSV2=r'./UCRArchive_2018/'+dataset_name+'/'+dataset_name+r'_TEST.tsv'
    train_dataset = readucr(filenameTSV1)
    test_dataset = readucr(filenameTSV2)
    return train_dataset, test_dataset

def get_data_loaders(dataset_name='ECG5000',batch_size=64):
    train_dataset, test_dataset = read(dataset_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, drop_last=False, shuffle=False)
    return train_loader, test_loader, train_dataset.shape[1]-1

def get_train_data_loaders(dataset_name='ECG5000',batch_size=64):
    train_dataset, test_dataset = read(dataset_name)
    train_dataset = np.concatenate((train_dataset, test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    return train_loader, train_dataset.shape[1]-1

def get_test_data(dataset_name='ECG5000'):
    train_dataset, test_dataset = read(dataset_name)
    return test_dataset

def get_mts_train_data(dataset_name='PeMSD8'):
    if dataset_name == 'PeMSBay':
        dataset = np.array(pd.read_csv('./mts/'+dataset_name+'.csv', header=None).values.astype(float))
    else:
        dataset = np.array(pd.read_hdf('./mts/'+dataset_name+'.h5'))
    train_len = int(dataset.shape[0] * 0.7)
    train_dataset = torch.Tensor(dataset[:train_len,:])
    fea_num = dataset.shape[1]
    return train_dataset, fea_num

def get_mts_all_data(dataset_name='ECG5000'):
    filenameTSV1=r'./UCRArchive_2018/'+dataset_name+'/'+dataset_name+r'_TRAIN.tsv'
    filenameTSV2=r'./UCRArchive_2018/'+dataset_name+'/'+dataset_name+r'_TEST.tsv'
    train_dataset = readucr(filenameTSV1)
    test_dataset = readucr(filenameTSV2)
    train_dataset = np.concatenate((train_dataset, test_dataset))[:,1:]
    fea_num = train_dataset.shape[1]
    return train_dataset, fea_num

def get_mts_all_data_label(dataset_name='ECG5000'):
    filenameTSV1=r'./UCRArchive_2018/'+dataset_name+'/'+dataset_name+r'_TRAIN.tsv'
    filenameTSV2=r'./UCRArchive_2018/'+dataset_name+'/'+dataset_name+r'_TEST.tsv'
    train_dataset = readucr(filenameTSV1)
    test_dataset = readucr(filenameTSV2)
    train_dataset_label = np.concatenate((train_dataset, test_dataset))[:,0]
    return np.expand_dims(train_dataset_label,1)


def load_UEA(dataset):
    train_data = loadarff(f'./Multivariate_arff/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'./Multivariate_arff/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    get_mts_train_data()