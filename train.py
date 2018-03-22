# purpose: kerasによる花の画像を利用したCNNのテスト　学習編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 
# created: 2018-02-17
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


#data_format = "channels_first"
data_format = "channels_last"

PATH_TO_TRAIN_IMAGES = os.path.join('..\data', 'processed', 'train_images')
PATH_TO_TRAIN_DATA = os.path.join('..\data', 'given', 'train_master.tsv')
PATH_TO_MODELPARAM = os.path.join('..\models', 'param')
PATH_TO_MODEL = '..\models'

def load_train_data(path_to_train_images, path_to_train_data):
    print('loading train data ...')
    data = pd.read_csv(path_to_train_data, sep='\t')
    image_list = []
    X = []
    y = []
    for row in data.iterrows():
        f, l = row[1]['file_name'], row[1]['category_id']
        try:
            #im = Image.open(os.path.join(path_to_train_images, f))
            # you may write preprocess method here given an image
            # im = preprocess_methods.my_preprocess_method(im)
            #X.append(np.array(im).flatten())

            # 画像を32x32pixelに変換し、1要素が[R,G,B]3要素を含む２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            path = os.path.join(path_to_train_images, f)
            im = Image.open(path).resize((32, 32))
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

    print(X.shape)
    print(y)
    
    y = np_utils.to_categorical(y, 55)                   # ベルをone-hot-encoding形式に変換

    print(y)

    print('loading train data ...done.')
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

    # model.add(Activation('relu')) 活性化層（発火を判定するための層）。前後の層をreluという活性化関数でつなぐ

    # model.add(Dropout(0.25))
    #   過学習を防ぐためにある（逆から計算する際にどれだけ学習が固定される、
    #   ネットワークから切り離されるわけではない、層を深くするのが簡単になる）

    model.add(Conv2D(64, (3, 3), padding="same", data_format=data_format, input_shape=in_shape))  
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))

    # ２次元から１次元ベクトルに変換する
    model.add(Flatten())

    # ユニット数を減らす 
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(55))                # 出力層のユニット数は2

    # お約束。シグモイド関数を使う（0-1にして、softmax層を挟みたいため）
    model.add(Activation('sigmoid'))

    # 学習には寄与しないが、人間のためにある。
    # 例えば、タコとイカを判定した際に、たこ0.6 イカ0.55という結果になった時に信頼できないということを表したい。
    # 全体の比率を表してくれる。どのような認識でも、シグモイド関数とソフトマックス層を挟む
    model.add(Activation('softmax'))

    # 最適化器。誤差を元に係数を修正するためのやり方。
    # いろいろやり方がある。収束が早い方法(Adam)と汎化が良い方法(SGD)
    # 最初はAdamで収束したらSGD
    opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 最適化器のセット。lrは学習係数

    # 損失関数を指定している（categorical_crossentropy）
    # 識別問題の場合は、これ（categorical_crossentropy）結果をどうやって1と0にするか
    # metricsは学習には寄与しない。人間のため
    model.compile(optimizer=opt,       # コンパイル
          loss='categorical_crossentropy',
          metrics=['accuracy'])

    print(model.summary())

    print('create the model ...done.')
    return model


def train_model(model, X_train, Y_train):
    print('train the model ...')

    # 学習
    # 1つのデータ当たりの学習回数
    # ある程度までは多ければいいが、メモリを食う。多すぎると正しい判定にならない（人生相談に言い換えるとわかりやすいかもしれない）
    epochs = 20
    # 学習係数を更新するために使う教師データ数
    batch_size = 100

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

    train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size)

    test_generator = datagen.flow(X_train, Y_train, batch_size=batch_size)

    checkpointer = ModelCheckpoint(filepath=PATH_TO_MODELPARAM+'\model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(PATH_TO_MODEL + '\model.log')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                      patience=5, min_lr=0.001)

    BATCH_SIZE = 16

    NUM_TRAINING = 160 #1600
    NUM_VALIDATION = 64 #400

    EPOCHS = 10

    # リアルタイムに水増し生成されるバッチ画像に対するモデルの適用
    history = model.fit_generator(train_generator,
                    steps_per_epoch=NUM_TRAINING/BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=test_generator,
                    validation_steps=NUM_VALIDATION/BATCH_SIZE,
                    verbose=1,
                    callbacks=[reduce_lr, csv_logger, checkpointer])

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


## load the data for training
X_train, Y_train = load_train_data(PATH_TO_TRAIN_IMAGES, PATH_TO_TRAIN_DATA)
    
## instanciate the model
model = instanciate_model(X_train)

## train the model
model = train_model(model, X_train, Y_train)

## save the trained model
save_model(model, PATH_TO_MODEL, PATH_TO_MODELPARAM)



