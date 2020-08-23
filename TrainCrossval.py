from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from networks import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, cohen_kappa_score, confusion_matrix

from utils import load_data, AA_andEachClassAccuracy

import keras.backend as k

k.set_image_data_format('channels_last')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LOAD DATASET
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Select dataset. Options are 'Kochia', 'IP', 'PU', 'SA', or 'EUROSAT'
dataset = 'SA'
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
TRAIN PROPOSED NETWORK
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
    ytrain = to_categorical(train_y[train]).astype(np.int32)
    ytest = to_categorical(train_y[test]).astype(np.int32)

    # Compile model
    model = hyper3dnet(img_shape=(windowSize, windowSize, train_x.shape[3], 1), classes=classes)
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.trainable = True

    # checkpoint
    filepath = "weights/" + dataset + "/" + network + "/weights-" + network + dataset + str(ntrain) + ".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    ep = 50
    # Train model on dataset
    print(dataset + ": Training" + str(ntrain) + "begins...")
    history = model.fit(x=train_x[train], y=ytrain, validation_data=(train_x[test], ytest),
                        batch_size=4, epochs=ep, callbacks=callbacks_list)

    # Evaluate network
    model.trainable = False
    model.load_weights(filepath)
    ypred = model.predict(train_x[test])

    # Calculate metrics
    oa = accuracy_score(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    confusion = confusion_matrix(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(ytest, axis=1), np.argmax(ypred, axis=-1))
    prec, rec, f1, support = precision_recall_fscore_support(np.argmax(ytest, axis=1),
                                                             np.argmax(ypred, axis=-1), average='macro')

    # Add metrics to the list
    cvoa.append(oa * 100)
    cvaa.append(aa * 100)
    cvka.append(kappa * 100)
    cvpre.append(prec * 100)
    cvrec.append(rec * 100)
    cvf1.append(f1 * 100)

    ntrain += 1
