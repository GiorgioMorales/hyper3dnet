import os
import scipy.io as sio
import numpy as np
import h5py
import pickle
from operator import truediv
from sklearn.decomposition import PCA


def AA_andEachClassAccuracy(confusion_m):
    list_diag = np.diag(confusion_m)
    list_raw_sum = np.sum(confusion_m, axis=1)
    each_ac = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_ac)
    return each_ac, average_acc


def applyPCA(Xc, numComponents=75):
    newX = np.reshape(Xc, (-1, Xc.shape[2]))
    pcaC = PCA(n_components=numComponents, whiten=True)
    newX = pcaC.fit_transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[1], numComponents))
    return newX, pcaC


def applyPCAtest(Xc, pcaC, numComponents=75):
    newX = np.reshape(Xc, (-1, Xc.shape[2]))
    newX = pcaC.transform(newX)
    newX = np.reshape(newX, (Xc.shape[0], Xc.shape[1], numComponents))
    return newX


def load_data(dataset="", ann=False, test=True):
    """Load specified dataset"""
    x = None
    y = None

    if dataset == "Kochia":
        x, y = load_Kochia(ann, test)
    elif dataset == "IP" or dataset == "PU" or dataset == "SA":
        x, y = load_HSISAT(dataset)
        x, pca = applyPCA(x, numComponents=30)
        x, y = createImageCubes(x, y, window=25)
    elif dataset == 'EUROSAT':
        x, y = load_EUROSAT()

    return x, y


def load_Kochia(ann=False, test=True):
    """Load Kochia dataset"""

    # Download the dataset from https://montana.box.com/s/mhpi7mxlw68abb616v0zl9t03zfwue63
    # and paste it in the project folder

    hdf5_file = h5py.File('weed_dataset_w25.hdf5', "r")
    train_x = np.array(hdf5_file["train_img"][...])
    train_y = np.array(hdf5_file["train_labels"][...])

    if ann:
        # Here, the pre-processing step for KochiaFC is different. See Sec. 5.
        train_x = train_x[:, 8:18, 8:18, :, :]
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2], train_x.shape[3]))
        listx = []
        listy = []
        for i in range(0, len(train_x)):
            for j in range(0, train_x.shape[1]):
                nir = np.mean(train_x[i, j, 175:199])
                red = np.mean(train_x[i, j, 132:156])
                ndvi = (nir - red) / (nir + red)
                if ndvi > 0.6:
                    listx.append(train_x[i, j, :])
                    listy.append(train_y[i])
        train_x = np.array(listx)
        train_y = np.array(listy)

    else:
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3], 1))

        cant = train_x.shape[0]
        size = train_x.shape[1]
        nband = train_x.shape[3]
        print("PCA ransformation begins")
        if test:
            # Load saved PCA transformation
            with open('Data/pca_Kochia', 'rb') as f:
                pca = pickle.load(f)
            train_x = applyPCAtest(np.reshape(train_x, (cant * size, size, nband)), pca, numComponents=100)
        else:
            train_x, pca = applyPCA(np.reshape(train_x, (cant * size, size, nband)), numComponents=100)
            with open('Data/pca_Kochia', 'wb') as f:
                pickle.dump(pca, f)
        # Reshape to its original window shape
        train_x = np.reshape(train_x, (cant, size, size, 100, 1))
        train_x = (train_x - np.mean(train_x)) / np.std(train_x)

    return train_x, train_y


def load_EUROSAT():
    """Load EUROSAT dataset"""

    # The original EUROSAT dataset can be downloaded from https://github.com/phelber/EuroSAT. Alternatively, a
    # pre-processed ready-to-use dataset that combines all the images in a single ".h5" file can be downloaded from
    # https://montana.box.com/s/wqakb91vp3fwe272ctx88n791s4gnqvj

    hdf5_file = h5py.File('EUROSAT.hdf5', "r")
    train_x = np.array(hdf5_file["train_img"][...]).astype(np.float) / 4000
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], train_x.shape[2], train_x.shape[3]))
    train_y = np.array(hdf5_file["train_labels"][...])
    test_x = np.array(hdf5_file["test_img"][...]).astype(np.float) / 4000
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], test_x.shape[2], test_x.shape[3]))
    test_y = np.array(hdf5_file["test_labels"][...])

    train_x = np.concatenate((train_x, test_x))
    train_y = np.concatenate((train_y, test_y))

    return train_x, train_y


def load_HSISAT(name):
    data_path = os.path.join(os.getcwd(), 'Data')
    if name == 'IP':
        dat = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        label = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        return dat, label
    elif name == 'SA':
        dat = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        label = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        return dat, label
    elif name == 'PU':
        dat = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        label = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        return dat, label


def padWithZeros(Xc, margin=2):
    newX = np.zeros((Xc.shape[0] + 2 * margin, Xc.shape[1] + 2 * margin, Xc.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:Xc.shape[0] + x_offset, y_offset:Xc.shape[1] + y_offset, :] = Xc
    return newX


def createImageCubes(Xc, yc, window=5, removeZeroLabels=True):
    margin = int((window - 1) / 2)
    zeroPaddedX = padWithZeros(Xc, margin=margin)
    # split patches
    patchesData = np.zeros((Xc.shape[0] * Xc.shape[1], window, window, Xc.shape[2]))
    patchesLabels = np.zeros((Xc.shape[0] * Xc.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = yc[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def Patch(data, height_index, width_index, windowSize):
    height_slice = slice(height_index, height_index + windowSize)
    width_slice = slice(width_index, width_index + windowSize)
    patch = data[height_slice, width_slice, :]

    return patch