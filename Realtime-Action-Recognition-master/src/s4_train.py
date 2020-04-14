#!/usr/bin/env python
# coding: utf-8

''' This script does:
1. Load features and labels from csv files
2. Train the model
3. Save the model to `model/` folder.
'''

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, BatchNormalization
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_classifier import ClassifierOfflineTrain



def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s4_train.py"]

CLASSES = np.array(cfg_all["classes"])


SRC_PROCESSED_FEATURES = par(cfg["input"]["processed_features"])
SRC_PROCESSED_FEATURES_LABELS = par(cfg["input"]["processed_features_labels"])

DST_MODEL_PATH = par(cfg["output"]["model_path"])

# -- Functions

def train_test_split(X, Y, ratio_of_test_size):
    ''' Split training data by ratio '''
    IS_SPLIT_BY_SKLEARN_FUNC = True

    # Use sklearn.train_test_split
    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 1
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)

    # Make train/test the same.
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y
def train(model, X, Y):
    ''' Train model. The result is saved into self.clf '''
    NUM_FEATURES_FROM_PCA = 100
    n_components = min(NUM_FEATURES_FROM_PCA, X.shape[1])
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X)
    # print("Sum eig values:", np.sum(self.pca.singular_values_))
    print("Sum eig values:", np.sum(pca.explained_variance_ratio_))
    X_new = pca.transform(X)
    print("After PCA, X.shape = ", X_new.shape)


    return model 

def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):
    ''' Evaluate accuracy and time cost '''

    # Accuracy
    t0 = time.time()

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    # Time cost
    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: "
          "{:.5f} seconds".format(average_time))

    # Plot accuracy
    axis, cf = lib_plot.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()



# -- Main
def conv_model(in_shape):
    num_classes = 9
    in_ly = Input(shape=(314,1))
    conv1 = Conv1D(32, 4, activation='relu')(in_ly)
    conv2 = Conv1D(32, 4, activation='relu')(conv1)
    dense1 = Dense(32, activation='relu')(conv2)
    # batch_norm1 = BatchNormalization()(conv2)
    max_p1 = MaxPooling1D(2)(dense1)
    conv3 = Conv1D(64, 4, activation='relu')(max_p1)
    conv4 = Conv1D(64, 4, activation='relu')(conv3)
    dense2 = Dense(64, activation='relu')(conv4)

    max_p2 = MaxPooling1D(2)(dense2)
    conv5 = Conv1D(128, 4, activation='relu')(max_p2)
    conv6 = Conv1D(128, 4, activation='relu')(conv5)
    dense3 = Dense(128, activation='relu')(conv6)
    # batch_norm2 = BatchNormalization()(conv4)
    g_pool = GlobalAveragePooling1D()(dense3)
    # drop = Dropout(0.5)(conv2)

    predictions = Dense(num_classes, activation='softmax')(g_pool)
    model = Model(inputs=in_ly, outputs=predictions)
    print(model.summary())
    return model

def main():
    

    # -- Load preprocessed data
    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(SRC_PROCESSED_FEATURES, dtype=float)  # features
    Y = np.loadtxt(SRC_PROCESSED_FEATURES_LABELS, dtype=int)  # labels
    
    # -- Train-test split
    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.2)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", np.unique(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    # -- Train the model
    print("\nStart training model ...")
    model = conv_model(tr_X.shape)
    # model = ClassifierOfflineTrain(tr_X.shape, )

    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(tr_X)
    # tr_X = imp.transform(tr_X)
    where_are_NaNs = np.isnan(tr_X)
    tr_X[where_are_NaNs] = 0
    where_are_neginf = np.isneginf(tr_X)
    tr_X[where_are_neginf] = 0
    where_are_inf = np.isinf(tr_X)
    tr_X[where_are_inf] = 0
    for i in tr_X:
        if np.isinf(i).any():
            print(i)

    callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='pose_classification.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8)
    ]

    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

    BATCH_SIZE = 128
    EPOCHS = 100
    tr_X = np.expand_dims(tr_X, axis=2)

    history = model.fit(tr_X,
              tr_Y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=callbacks_list,
              validation_split=0.2,
              verbose=1)
    # model.fit(tr_X, tr_Y)
    # history = train(model , tr_X, tr_Y)
    # -- Evaluate model
    print("\nStart evaluating model ...")
    # where_are_NaNs = np.isnan(te_X)
    # te_X[where_are_NaNs] = 0
    # evaluate_model(model, CLASSES, tr_X, tr_Y, te_X, te_Y)

    # -- Save model
    # print("\nSave model to " + DST_MODEL_PATH)
    # with open(DST_MODEL_PATH, 'wb') as f:
    #     pickle.dump(model, f)


if __name__ == "__main__":
    main()
