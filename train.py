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


#data_format = "channels_first"
data_format = "channels_last"

PATH_TO_TRAIN_IMAGES = os.path.join('..\data', 'processed', 'train_images')
PATH_TO_TRAIN_DATA = os.path.join('..\data', 'given', 'train_master.tsv')
PATH_TO_MODEL = os.path.join('models', 'param')

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

#def openfile(dir_name, data_format="channels_last"):
#    """ 画像をリストとして返す
#    """
#    image_list = []
#    files = os.listdir(dir_name)    # ディレクトリ内部のファイル一覧を取得
#    print(files)
#    for file in files:
#        root, ext = os.path.splitext(file)  # 拡張子を取得
#        if ext != ".jpg":
#            continue
#        path = os.path.join(dir_name, file) # ディレクトリ名とファイル名を結合して、パスを作成
#        # 画像を32x32pixelに変換し、1要素が[R,G,B]3要素を含む２次元配列として読み込む。
#        # [R,G,B]はそれぞれが0-255の配列。
#        image = np.array(Image.open(path).resize((32, 32)))
#        if data_format == "channels_first":
#            image = image.transpose(2, 0, 1)   # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
#        image = image / 255.                   # 値を0-1に正規化
#        image_list.append(image)        # 出来上がった配列をimage_listに追加  
#    return image_list
## 画像を読み込む
#img1 = openfile('1_train')
#img2 = openfile('2_train')
#x = np.array(img1 + img2)  # リストを結合
#y = np.array([0] * len(img1) + [1] * len(img2))  # 正解ラベルを作成
#y = np_utils.to_categorical(y)                   # ベルをone-hot-encoding形式に変換
#print(x.shape)
#print(y)
##exit()

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


def instanciate_model(X):
    print('create the model ...')

    # input_shapeに32:32:3が渡される形。縦、横、RGB
    in_shape = X.shape[1:]

    # モデルの作成
    model = Sequential()

    # 畳み込み層の作成
    # 1層目の追加2次元の層の畳み込み、padding=same,畳み込みしても画像サイズが変わらないように
    # カーネル数32, カーネルサイズ(3,3), input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d
    model.add(Conv2D(32, (3, 3), padding="same", data_format=data_format, input_shape=in_shape))  
    # 活性化層（発火を判定するための層）。前後の層をreluという活性化関数でつなぐ
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 過学習を防ぐためにある（逆から計算する際にどれだけ学習が固定される、ネットワークから切り離されるわけではない、層を深くするのが簡単になる）
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
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

def train_model(model, X, y):
    print('train the model ...')

    # 学習
    epochs = 20 # 1つのデータ当たりの学習回数
    # ある程度までは多ければいいが、メモリを食う。多すぎると正しい判定にならない（人生相談に言い換えるとわかりやすいかもしれない）
    batch_size = 32              # 学習係数を更新するために使う教師データ数

    history = model.fit(X, y, 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=1,              # 学習に学習の経過を表示するかどうか
        validation_split=0.2,   # 渡された教師データのうち、何割を使用するか
        #validation_data=(x_test, y_test), # validation_dataをセットするとvalidation_splitは無視される
        shuffle=True,           # 学習毎にデータをシャッフルする。学習の汎化性能を上げる（同じ順番で学習すると学習に相当影響するため。ゲームで同じ順番だとあるパターンにしか対応できなくなるのと一緒）
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）


    # サマリー
    print(model.summary())                    # レイヤー情報を表示(上で表示させると流れるので)

    # モデルの評価
    print('evalute.')
    score = model.evaluate(X, y)
    print('')
    print('evalute...done.')

    print("test loss", score[0])
    print("test acc", score[1])

    #プロット
    plot_history(history)                     # lossの変化をグラフで表示

    print('training the model ...done.')

def save_model(model, name):
    print('saving the model ...')

    # 学習結果を保存 次の判定に使用する。モデルと結合係数は別々に保存することで、モデルは別の人を使用し、結合係数は自分のものを使用するということができる
    open("model", "w").write(model.to_json()) # モデル情報の保存
    model.save_weights(name+'.hdf5')          # 獲得した結合係数を保存

    print('saving the model ...done.')


## load the data for training
X, y = load_train_data(PATH_TO_TRAIN_IMAGES, PATH_TO_TRAIN_DATA)
    
## instanciate the model
model = instanciate_model(X)

## train the model
model = train_model(model, X, y)

## save the trained model
save_model(model, PATH_TO_MODEL)



