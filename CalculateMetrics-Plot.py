from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import spectral
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix

from utils import load_data, AA_andEachClassAccuracy, load_HSISAT, Patch, applyPCA, padWithZeros

import keras.backend as k
import tensorflow as tf
k.set_image_data_format('channels_last')
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Uncomment to use CPU instead o GPU


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD DATASET
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Select dataset. Options are 'Kochia', 'IP', 'PU', 'SA', or 'EUROSAT'
dataset = 'IP'
train_x, train_y = load_data(dataset=dataset, test=True)
if dataset == 'Kochia':
    classes = 3
elif dataset == 'EUROSAT':
    classes = 10
else:
    classes = int(np.max(train_y)) + 1
print("Dataset correctly imported")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
EVALUATION
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

windowSize = train_x.shape[1]
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvoa = []
cvaa = []
cvka = []
cvpre = []
cvrec = []
cvf1 = []

# Select Network. Options are 'hyper3dnet', 'hybridsn', 'spectrum',
# 'resnet50' (expect for Kochia), or 'kochiafc' (only for Kochia)
network = 'hyper3dnet'

# Initialize
confmatrices = np.zeros((10, int(classes), int(classes)))

ntrain = 1
model = None
for train, test in kfold.split(train_x, train_y):

    k.clear_session()

    ytest = to_categorical(train_y[test], num_classes=classes).astype(np.int32)

    # Load network and weights of the 'ntrain'-fold
    model = load_model("weights/" + dataset + "/" + network + "/weights-" + network + dataset + str(ntrain) + ".h5")
    model.trainable = False

    # Predict results
    ypred = model.predict(train_x[test])

    # Calculate confusion matrix
    sess = tf.compat.v1.Session()
    with sess.as_default():
        con_mat = tf.math.confusion_matrix(labels=np.argmax(ytest, axis=-1),
                                           predictions=np.argmax(ypred, axis=-1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=3)
    classes_list = list(range(0, int(classes)))
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes_list, columns=classes_list)
    confmatrices[ntrain - 1, :, :] = con_mat_df.values

    # Calculate metrics
    oa = accuracy_score(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    confusion = confusion_matrix(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    prec, rec, f1, support = precision_recall_fscore_support(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1),
                                                             average='macro')

    # Add metrics to the list
    cvoa.append(oa * 100)
    cvaa.append(aa * 100)
    cvka.append(kappa * 100)
    cvpre.append(prec * 100)
    cvrec.append(rec * 100)
    cvf1.append(f1 * 100)

    ntrain = ntrain + 1

# Selects the fold with the minimum performance
nmin = np.argmin(cvoa)
model = load_model("weights/" + dataset + "/" + network + "/weights-" + network + dataset + str(nmin + 1) + ".h5")
model.trainable = False

file_name = "classification_report_hyper3dnet" + dataset + ".txt"
with open(file_name, 'w') as x_file:
    x_file.write("Overall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvoa)), float(np.std(cvoa))))
    x_file.write('\n')
    x_file.write("Average accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvaa)), float(np.std(cvaa))))
    x_file.write('\n')
    x_file.write("Kappa accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvka)), float(np.std(cvka))))
    x_file.write('\n')
    x_file.write("Precision accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvpre)), float(np.std(cvpre))))
    x_file.write('\n')
    x_file.write("Recall accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvrec)), float(np.std(cvrec))))
    x_file.write('\n')
    x_file.write("F1 accuracy%.3f%% (+/- %.3f%%)" % (float(np.mean(cvf1)), float(np.std(cvf1))))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PLOT
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Segmentation result
if dataset == 'IP' or dataset == 'PU' or dataset == 'SA':
    X, y = load_HSISAT(dataset)
    height = y.shape[0]
    width = y.shape[1]
    X, pca = applyPCA(X, numComponents=30)
    X = padWithZeros(X, windowSize // 2)

    # calculate the predicted image
    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                image_patch = Patch(X, i, j, windowSize)
                print(i, " ", j)
                X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1],
                                                   image_patch.shape[2]).astype('float32')
                prediction = (model.predict(X_test_image))
                prediction = np.argmax(prediction, axis=1)
                outputs[i][j] = prediction + 1

    # Print Ground-truth, network output, and comparison
    colors = spectral.spy_colors
    colors[1] = [125, 80, 0]
    colors[2] = [80, 125, 0]
    colors[4] = [255, 0, 0]
    colors[10] = [150, 30, 100]
    colors[11] = [200, 100,   0]
    ground_truth = spectral.imshow(classes=y, figsize=(7, 7), colors=colors)
    spectral.save_rgb('weights/' + dataset + network + 'dataset_gt.png', y, colors=colors)
    predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(7, 7), colors=colors)
    spectral.save_rgb('weights/' + dataset + network + 'dataset_out.png', outputs.astype(int), colors=colors)
    outrgb = cv2.imread('weights/' + dataset + network + 'dataset_out.png', cv2.IMREAD_COLOR)
    outrgb = cv2.cvtColor(outrgb, cv2.COLOR_BGR2RGB)

    for i in range(height):
        for j in range(width):
            if y[i, j] != outputs.astype(int)[i, j]:
                outrgb[i, j] = [255, 255, 0]  # Mark the errors in yellow

    plt.figure()
    plt.imshow(outrgb)
