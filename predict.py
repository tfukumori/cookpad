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


#data_format = "channels_first"
data_format = "channels_last"

PATH_TO_TEST_IMAGES = os.path.join('..\data', 'processed', 'test_images')
PATH_TO_TEST_PICKLE_X = os.path.join('..\data', 'processed', 'test_images_edit_X.pickle')
PATH_TO_MODEL = '..\models'
PATH_TO_MODELPARAM = os.path.join('..\models', 'param')
PATH_TO_SUBMIT_FILE = 'submit.csv'

IMAGE_WIDTH=64
IMAGE_HEIGHT=64

def load_test_data(path_to_test_images, path_to_images_pickle_x):
    print('loading test data ...')
    image_list = []
    X = []
    file_name = []
    file = os.listdir(path_to_test_images)

    if os.path.isfile(path_to_images_pickle_x):
        with open(path_to_images_pickle_x, 'rb') as f:
            X = pickle.load(f)

    else:
        for f in file:
            try:
                im = Image.open(os.path.join(PATH_TO_TEST_IMAGES, f)).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
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

    print(X.shape)

    print('loading data ...done ' + path_to_test_images)
    return X, file_name

def predict(model, X, file_name):
    print('predicting ...')
    dic = OrderedDict()
    dic['file_name'] = file_name
    dic['predict,ion'] = model.predict(
        X,
        batch_size=None,
        vervose=1
    )
    print('done.')
    return pd.DataFrame(dic)

X, file_name = load_data(PATH_TO_TEST_IMAGES, PATH_TO_TEST_PICKLE_X)

# 機械学習器を復元
model = model_from_json(open(PATH_TO_MODEL +"\model", 'r').read())
model.load_weights(PATH_TO_MODELPARAM + "\model.hdf5")

## output the submit file
submit = predict(model, X, file_name)
submit.to_csv(PATH_TO_SUBMIT_FILE, index=None, header=None)