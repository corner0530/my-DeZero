# 58. 代表的なCNN(VGG16)
import numpy as np
from PIL import Image

import dezero
from dezero.models import VGG16

if __name__ == "__main__":
    url = (
        "https://github.com/oreilly-japan/deep-learning-from-scratch-3/"
        "raw/images/zebra.jpg"
    )
    img_path = dezero.utils.get_file(url)
    img = Image.open(img_path)

    x = VGG16.preprocess(img)
    x = x[np.newaxis]  # バッチ用の軸を追加

    model = VGG16(pretrained=True)
    with dezero.test_mode():
        y = model(x)
    predict_id = np.argmax(y.data)

    model.plot(x, to_file="vgg.pdf")  # 計算グラフの可視化
    labels = dezero.datasets.ImageNet.labels()  # ImageNetのラベル
    print(labels[predict_id])
