import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from image_fit import resize_to_fit


LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# 初始化数据和标签
data = []
labels = []

# 循环输入图像
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # 加载图像并将其转换为灰度
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调整字母大小，使其适合20x20像素的框
    image = resize_to_fit(image, 20, 20)

    # 为图像添加第三个通道维度以适配Keras
    image = np.expand_dims(image, axis=2)

    # 根据字母所在的文件夹获取其名称
    label = image_file.split(os.path.sep)[-2]

    # 在我们的训练数据中添加字母图像及其标签
    data.append(image)
    labels.append(label)


# 将原始像素强度缩放到范围[0，1]（这将改进训练）
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 将训练数据分为单独的训练集和测试集
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.1,  random_state=0)

# 将标签（字母）转换为Keras可以使用的独热编码
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# 保存从标签到独热编码的映射。当我们使用模型来解码它的预测意味着什么时，我们需要这个
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# 建立神经网络
model = Sequential()

# 最大池的第一卷积层
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 最大池的第二卷积层
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 500节点隐藏层
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# 具有10个节点的输出层（我们预测的每个可能的字母/数字对应一个节点）
model.add(Dense(10, activation="softmax"))

# Keras建立TensorFlow模型
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练神经网络
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=25, verbose=1)

# 将训练过的模型保存到磁盘
model.save(MODEL_FILENAME)
