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

PATH_TO_TRAIN_IMAGES = os.path.join('..\data', 'processed', 'train_images')
PATH_TO_TRAIN_DATA = os.path.join('..\data', 'given', 'train_master.tsv')
PATH_TO_TRAIN_PICKLE_X = os.path.join('..\data', 'processed', 'train_images_edit_X.pickle')
PATH_TO_TRAIN_PICKLE_Y = os.path.join('..\data', 'processed', 'train_images_edit_y.pickle')
PATH_TO_TEST_IMAGES = os.path.join('..\data', 'processed', 'train_images')
PATH_TO_TEST_DATA = os.path.join('..\data', 'given', 'test_master.tsv')
PATH_TO_TEST_PICKLE_X = os.path.join('..\data', 'processed', 'test_images_edit_X.pickle')
PATH_TO_TEST_PICKLE_Y = os.path.join('..\data', 'processed', 'test_images_edit_y.pickle')
PATH_TO_MODEL = '..\models'
PATH_TO_MODELPARAM = os.path.join('..\models', 'param')
PATH_TO_MODELPARAM_CONTINUE = os.path.join(PATH_TO_MODELPARAM, 'model.continue.hdf5')

IMAGE_WIDTH=64
IMAGE_HEIGHT=64

def load_data(path_to_images, path_to_data, path_to_images_pickle_x, path_to_images_pickle_y):
    print('loading data ...' + path_to_data)
    data = pd.read_csv(path_to_data, sep='\t')
    image_list = []
    X = []
    y = []
    if os.path.isfile(path_to_images_pickle_x):
        with open(path_to_images_pickle_x, 'rb') as f:
            X = pickle.load(f)
        with open(path_to_images_pickle_y, 'rb') as f:
            y = pickle.load(f)

    else:
        for row in data.iterrows():
            f, l = row[1]['file_name'], row[1]['category_id']
            try:
                #im = Image.open(os.path.join(path_to_train_images, f))
                # you may write preprocess method here given an image
                # im = preprocess_methods.my_preprocess_method(im)
                #X.append(np.array(im).flatten())

                # 画像を32x32pixelに変換し、1要素が[R,G,B]3要素を含む２次元配列として読み込む。
                # [R,G,B]はそれぞれが0-255の配列。
                path = os.path.join(path_to_images, f)
                im = Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
                image = np.array(im)
                if data_format == "channels_first":
                    image = image.transpose(2, 0, 1)   # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
                image = image / 255.                   # 値を0-1に正規化
                image_list.append(image)                  # 出来上がった配列を追加  

                y.append(l)
            except Exception as e:
                print(str(e))

        X = np.array(image_list)
        y = np.array(y)

        y = np_utils.to_categorical(y, 55)                   # one-hot-encoding形式に変換

        with open(path_to_images_pickle_x, 'wb') as f:
            pickle.dump(X, f)
        with open(path_to_images_pickle_y, 'wb') as f:
            pickle.dump(y, f)

    print(X.shape)
    print(y)

    print('loading data ...done ' + path_to_data)
    return X, y

def plot_history(history):
    print('plot history ...')

    """ 損失の履歴を図示する
    from http://www.procrasist.com/entry/2017/01/07/154441
    """
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()

    print('plot history ...done.')

def instanciate_model(X_train):
    print('create the model ...')

    # input_shapeに32:32:3が渡される形。縦、横、RGB
    in_shape = X_train.shape[1:]

    # モデルの作成
    model = Sequential()

    # 畳み込み層の作成

    # model.add(Conv2D(32, (3, 3))) 2次元の層の畳み込み
    #   カーネル数32, カーネルサイズ(3,3)
    #   padding=same,畳み込みしても画像サイズが変わらないように
    #   input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d

    # model.add(Activation('relu')) 
    # 　活性化層（発火を判定するための層）。前後の層をreluという活性化関数でつなぐ

    # model.add(Dropout(0.25))
    #   過学習を防ぐためにある（次のニューロンへのパスをランダムに調整する）
    #   ネットワークから切り離されるわけではない、層を深くするのが簡単になる）

    # model.add(Flatten())
    # 　２次元から１次元ベクトルに変換する（平坦化）

    # model.add(Activation('sigmoid'))
    # 　お約束。シグモイド関数を使う（0-1にして、softmax層を挟みたいため）

    # model.add(Activation('softmax'))
    # 　学習には寄与しないが、人間のためにある。
    # 　例えば、タコとイカを判定した際に、たこ0.6 イカ0.55という結果になった時に信頼できないということを表したい。
    # 　全体の比率を表してくれる。どのような認識でも、シグモイド関数とソフトマックス層を挟む

    #畳み込み層の追加
    #畳み込み層
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu', data_format=data_format, input_shape=in_shape))  
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #畳み込み層
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))  
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #畳み込み層
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))  
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    #畳み込み層
    model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))  

    # 平坦化
    model.add(Flatten())
    # 全結合層
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    # 出力層のユニット数
    model.add(Dense(55, activation='softmax'))                

    # ここまででモデルの層完成

    # 最適化器。
    # 誤差を元に係数を修正するためのやり方。
    # いろいろやり方がある。収束が早い方法(Adam)と汎化が良い方法(SGD)
    # 最初はAdamで収束したらSGD
    opt = keras.optimizers.Adam(
        lr=0.0005,              #学習係数
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0003
    ) 

    # コンパイル
    model.compile(
        optimizer=opt,                          
        loss='categorical_crossentropy',        # 損失関数を指定している（識別問題なら、categorical_crossentropy）
        metrics=['accuracy']                    # metricsは学習には寄与しない。人間のため
    )

    print(model.summary())

    print('create the model ...done.')
    return model

def load_model_fromfile(modelpath):
    print('load the model ...')
    return load_model(modelpath)

def train_model(model, X_train, Y_train, isValidation):
    print('train the model ...')

    BATCH_SIZE = 16
    if isValidation:
        EPOCHS = 100
    else:
        EPOCHS = 1000

    NUM_TRAINING = X_train.shape[0]
    RATE_TEST = 0.05
    NUM_TEST = int(NUM_TRAINING * RATE_TEST)
    RATE_VALIDATION = 0.20
    NUM_VALIDATION = int(NUM_TRAINING * RATE_VALIDATION)

    shuffle_dataset(X_train, Y_train)

    X_test = X_train[:NUM_TEST]
    Y_test = Y_train[:NUM_TEST]
    X_val = X_train[NUM_TEST:NUM_TEST+NUM_VALIDATION]
    Y_val = Y_train[NUM_TEST:NUM_TEST+NUM_VALIDATION]
    X_train = X_train[NUM_TEST:]
    Y_train = Y_train[NUM_TEST:]

    # データの水増し（Data Augmentation）
    datagen = ImageDataGenerator(      
                                 rotation_range=180,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    # 水増し画像を訓練用画像の形式に合わせる
    datagen.fit(X_train)

    # 過学習の抑制
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10 , verbose=1)

    # 評価に用いるモデル重みデータの保存
    #checkpointer = ModelCheckpoint(model_weights, monitor='val_loss', verbose=1, save_best_only=True)
    #callbacks=[early_stopping, checkpointer]) 

    if isValidation:
        train_generator = datagen.flow(X_val, Y_val, batch_size=BATCH_SIZE, shuffle=True)
        num_sample = NUM_VALIDATION
    else:
        train_generator = datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE, shuffle=True)
        num_sample = NUM_TRAINING

    test_generator = datagen.flow(X_test, Y_test, batch_size=BATCH_SIZE)

    # checkpointer1 = ModelCheckpoint(
    #         filepath=PATH_TO_MODELPARAM+'\model.{epoch:02d}-{val_loss:.2f}.hdf5',
    #         verbose=1,
    #         save_best_only=True
    #     )

    checkpointer2 = ModelCheckpoint(
            filepath=PATH_TO_MODELPARAM_CONTINUE,
            verbose=1,
            save_best_only=True,
            save_weights_only=False
        )

    csv_logger = CSVLogger(PATH_TO_MODEL + '\model.log')

    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.001
        )

    # リアルタイムに水増し生成されるバッチ画像に対するモデルの適用
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=num_sample/BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=test_generator,
            validation_steps=NUM_TEST/BATCH_SIZE,
            verbose=1,
            callbacks=[reduce_lr, csv_logger, checkpointer2] #, checkpointer1
        )

    #history = model.fit(X_train, Y_train, 
    #    epochs=epochs, 
    #    batch_size=batch_size, 
    #    verbose=1,                          # 学習に学習の経過を表示するかどうか
    #    validation_split=0.1,               # 渡された教師データのうち、何割を検証に使用するか
    #    #validation_data=(X_test, Y_test),  # validation_dataをセットするとvalidation_splitは無視される
    #    shuffle=True,                       # 学習毎にデータをシャッフルする。学習の汎化性能を上げる（同じ順番で学習すると学習に相当影響するため。ゲームで同じ順番だとあるパターンにしか対応できなくなるのと一緒）
    #    ) 
    
    # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）


    # サマリー
    print(model.summary())                    # レイヤー情報を表示(上で表示させると流れるので)

    # モデルの評価
    print('evalute.')
    if isValidation:
        score = model.evaluate(X_val, Y_val)
    else:
        score = model.evaluate(X_train, Y_train)
    print('')
    print('evalute...done.')

    print("test loss", score[0])
    print("test acc", score[1])

    #プロット
    plot_history(history)                     # lossの変化をグラフで表示

    print('training the model ...done.')
    return model

def save_model(model, modelpath, modelparampath):
    print('saving the model ...')

    #json_string = model.to_json()
    #if not os.path.isdir("cache"):
    #    os.mkdir("cache")
    #json_name = "architecture.json"
    #open(os.path.join("cache", json_name),"w").write(json_string)

    # 学習結果を保存 次の判定に使用する。モデルと結合係数は別々に保存することで、モデルは別の人を使用し、結合係数は自分のものを使用するということができる
    open(modelpath +"\model", "w").write(model.to_json()) # モデル情報の保存
    model.save_weights(modelparampath + "\model.hdf5")          # 獲得した結合係数を保存

    print('saving the model ...done.')

def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]
    return x, t

## load the data for training
X_train, Y_train = load_data(PATH_TO_TRAIN_IMAGES, PATH_TO_TRAIN_DATA, PATH_TO_TRAIN_PICKLE_X, PATH_TO_TRAIN_PICKLE_Y)

### load the data for training
#X_test, Y_test = load_data(PATH_TO_TEST_IMAGES, PATH_TO_TEST_DATA, PATH_TO_TEST_PICKLE_X, PATH_TO_TEST_PICKLE_Y)

if not os.path.isdir(PATH_TO_MODEL):
    os.mkdir(PATH_TO_MODEL)
    
if not os.path.isdir(PATH_TO_MODELPARAM):
    os.mkdir(PATH_TO_MODELPARAM)

if os.path.isfile(PATH_TO_MODELPARAM_CONTINUE):
    model = load_model_fromfile(PATH_TO_MODELPARAM_CONTINUE)
else:
    model = instanciate_model(X_train)

## train the model
model = train_model(model, X_train, Y_train, False)

## save the trained model
save_model(model, PATH_TO_MODEL, PATH_TO_MODELPARAM)



