#!/usr/bin/env python
# coding: utf-8

# In[62]:


import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, Flatten, Masking, Dropout, LSTM
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[110]:


df = pd.read_csv("/content/drive/MyDrive/ColabNotebookss/OpenPose/dataset_mp_F.csv")
df = df[df['label'].isin(['sitting', 'running', 'drinking', 'cycling', 'sleeping'])]
df.to_csv('data_set_mp_srdcs.csv', index=False)


# In[71]:


def split_dataset(x, y, train_split=0.8, validation_split=0.1, join_validation=True, keypoints_num=18):
    indices = np.array(range(len(x)))

    train_size = round(train_split * len(y))
    validation_size = round(validation_split * len(y))

    random.seed(123)
    # random.shuffle(indices)

    validation_indice_bound = train_size + validation_size
    train_indices = indices[0:train_size]
    validation_indices = indices[train_size:validation_indice_bound]
    test_indices = indices[validation_indice_bound:len(x)]

    x_train = x[train_indices, :].reshape(-1, keypoints_num, 2)
    x_val = x[validation_indices, :].reshape(-1, keypoints_num, 2)
    x_test = x[test_indices, :].reshape(-1, keypoints_num, 2)
    y_train = y[train_indices, :]
    y_val = y[validation_indices, :]
    y_test = y[test_indices, :]

    if join_validation:
        x_test = np.concatenate((x_test, x_val), axis=0)
        y_test = np.concatenate((y_test, y_val), axis=0)

    return x_train, y_train, x_val, y_val, x_test, y_test


# In[111]:


def load_dataset():
    df = pd.read_csv("/content/drive/MyDrive/ColabNotebookss/OpenPose/data_set_mp_srdcs.csv")
    if ('Unnamed: 0' in df.columns):
      df = df.drop('Unnamed: 0', axis=1)
    y_df = pd.get_dummies(df["label"])
    classes = y_df.columns
    x_df = df.drop("label", axis=1)

    return x_df.to_numpy(), y_df.to_numpy(), classes


# In[114]:


def get_data(join_validation, keypoints_num):
    data_x, data_y, classes = load_dataset()
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(data_x, data_y, join_validation=join_validation, keypoints_num=keypoints_num)
    return x_train, y_train, x_val, y_val, x_test, y_test, classes


# In[122]:


def train_nn(x_train, y_train, x_val, y_val, save_model=False, verbose=True, keypoints_num=18):
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(None, keypoints_num, 2), dtype="float32"))

    model.add(Flatten(data_format="channels_last"))

    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=16, validation_data = (x_val, y_val), epochs=200, verbose=verbose)


    if save_model:
        model.save("/content/drive/MyDrive/ColabNotebookss/OpenPose/HAR_model")

    predictions = tf.argmax(model.predict(x_val), axis=1).numpy()

    print(classification_report(tf.argmax(y_val, axis=1).numpy(), predictions))
    print(f1_score(tf.argmax(y_val, axis=1).numpy(), predictions, average="micro"))
    # print(predictions)


# In[123]:


def train_svm(x_train, y_train, x_val, y_val, verbose=True, save_model=False, keypoints_num=18):
    y_1d = tf.argmax(y_train, axis=1).numpy()
    x_train = x_train.reshape((x_train.shape[0], keypoints_num * 2))
    x_val = x_val.reshape((x_val.shape[0], keypoints_num * 2))

    clf = SVC(kernel="rbf", verbose=verbose)
    clf.fit(x_train, y_1d)
    predictions = clf.predict(x_val)

    print(classification_report(tf.argmax(y_val, axis=1).numpy(), predictions))
    print(f1_score(tf.argmax(y_val, axis=1).numpy(), predictions, average="micro"))

    if save_model:
        with open("svm.pickle", "wb") as file:
            pickle.dump(clf, file)


# In[124]:


def train_random_forest(x_train, y_train, x_val, y_val, verbose=True, save_model=False, keypoints_num=18):
    y_1d = tf.argmax(y_train, axis=1).numpy()
    x_train = x_train.reshape((x_train.shape[0], keypoints_num * 2))
    x_val = x_val.reshape((x_val.shape[0], keypoints_num * 2))

    clf = RandomForestClassifier(n_estimators=500, max_depth=100, verbose=verbose, n_jobs=-1, random_state=123)
    clf.fit(x_train, y_1d)
    predictions = clf.predict(x_val)

    print(classification_report(tf.argmax(y_val, axis=1).numpy(), predictions))
    print(f1_score(tf.argmax(y_val, axis=1).numpy(), predictions, average="micro"))

    if save_model:
        with open("random_forest.pickle", "wb") as file:
            pickle.dump(clf, file)


# In[127]:


x_train, y_train, x_val, y_val, x_test, y_test, classes = get_data(False, 33)
# train_nn(x_train, y_train, x_test, y_test, save_model=True, verbose=False, keypoints_num=33)
# train_svm(x_train, y_train, x_test, y_test, save_model=True, verbose=False, keypoints_num=33)
train_random_forest(x_train, y_train, x_test, y_test, save_model=True, verbose=False, keypoints_num=33)
print(classes)


# In[ ]:




