# purpose: kerasによる花の画像を利用したCNNのテスト　学習編
# author: Katsuhiro MORISHITA　森下功啓
# edit: FUKUMORI
# memo: 
# created: 2018-02-17

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import os
import pickle
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from collections import OrderedDict


#data_format = "channels_first"
data_format = "channels_last"

PATH_TO_TEST_IMAGES = os.path.join('..\data', 'processed', 'test_images')
PATH_TO_TEST_PICKLE_X = os.path.join('..\data', 'processed', 'test_images_edit_X.pickle')
PATH_TO_TEST_PICKLE_FILENAME = os.path.join('..\data', 'processed', 'test_file_name.pickle')
PATH_TO_MODEL = '..\models'
PATH_TO_MODELPARAM = os.path.join('..\models', 'param')
PATH_TO_SUBMIT_FILE = 'submit.csv'

IMAGE_WIDTH=64
IMAGE_HEIGHT=64

def load_test_data(path_to_test_images, path_to_images_pickle_x, path_to_pickle_file_name):
    print('loading test data ...')
    image_list = []
    X = []
    file_name = []
    file = os.listdir(path_to_test_images)

    if os.path.isfile(path_to_images_pickle_x):
        with open(path_to_images_pickle_x, 'rb') as f:
            X = pickle.load(f)
        with open(path_to_pickle_file_name, 'rb') as f:
            file_name = pickle.load(f)

    else:
        for f in file:
            try:
                im = Image.open(os.path.join(PATH_TO_TEST_IMAGES, f)).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                # image_list.append(np.array(im).flatten())

                image = np.array(im)
                if data_format == "channels_first":
                    image = image.transpose(2, 0, 1)   # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
                image = image / 255.                   # 値を0-1に正規化
                image_list.append(image)                  # 出来上がった配列を追加  
                
                file_name.append(f)
            except Exception as e:
                print(str(e))

        X = np.array(image_list)

        with open(path_to_images_pickle_x, 'wb') as f:
            pickle.dump(X, f)
        with open(path_to_pickle_file_name, 'wb') as f:
            pickle.dump(file_name, f)

    print(X.shape)

    print('loading data ...done ' + path_to_test_images)
    return X, file_name

def predict(model, X, file_name):
    print('predicting ...')
    result = model.predict(
        X,
        None,
        1
    )

    print('predicting classes ...')
    resultClasses = model.predict_classes(
        X,
        None,
        1
    )

    print(str(file_name))
    print(str(result))
    print(str(resultClasses))

    df = pd.DataFrame()
    df['file_name'] = file_name
    # df['prediction'] = result
    df['prediction'] = resultClasses

    print('done.')
    return df

X, file_name = load_test_data(PATH_TO_TEST_IMAGES, PATH_TO_TEST_PICKLE_X, PATH_TO_TEST_PICKLE_FILENAME)

# 機械学習器を復元
print('load_model.')
model = load_model(PATH_TO_MODELPARAM +"\model.continue.hdf5")
# model = load_model(PATH_TO_MODEL +"\model")
# print('load_weights.')
# model.load_weights(PATH_TO_MODELPARAM + "\model.hdf5")

## output the submit file
submit = predict(model, X, file_name)
submit.to_csv(PATH_TO_SUBMIT_FILE, index=None, header=None)