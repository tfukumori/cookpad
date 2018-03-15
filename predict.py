
# purpose: kerasによる花の画像を利用したCNNのテスト　　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 
# created: 2018-02-17
from sklearn import preprocessing # 次元毎の正規化に使う
from keras.models import model_from_json
from PIL import Image
import numpy as np
import os



# データの読み込み
def openfile(dir_name, data_format="channels_last"):
    """ 画像をリストとして返す
    """
    image_list = []
    files = os.listdir(dir_name)    # ディレクトリ内部のファイル一覧を取得
    print(files)

    for file in files:
        root, ext = os.path.splitext(file)  # 拡張子を取得
        if ext != ".jpg":
            continue

        path = os.path.join(dir_name, file) # ディレクトリ名とファイル名を結合して、パスを作成
        # 画像を32x32pixelに変換し、1要素が[R,G,B]3要素を含む２次元配列として読み込む。
        # [R,G,B]はそれぞれが0-255の配列。
        image = np.array(Image.open(path).resize((32, 32)))
        if data_format == "channels_first":
            image = image.transpose(2, 0, 1)   # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        image = image / 255.                   # 値を0-1に正規化
        image_list.append(image / 255.)        # 出来上がった配列をimage_listに追加  
    
    return image_list

# 画像を読み込む
img1 = openfile('1_test')
img2 = openfile('2_test')
x = np.array(img1 + img2)  # リストを結合

# 機械学習器を復元
model = model_from_json(open('model', 'r').read())
model.load_weights('param.hdf5')


# テスト用のデータを保存
with open("test_result.csv", "w") as fw:
    test = model.predict_classes(x)
    test = [str(val) for val in test] # 出力を文字列に変換
    print(test)
    fw.write("{0}\n".format("\n".join(test)))