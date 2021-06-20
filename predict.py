from keras.models import load_model
from image_fit import resize_to_fit
from imutils import paths
import numpy as np
import imutils
import cv2
import pickle
import os
# import matplotlib.pyplot as plt


# 计算邻域非白色个数
def calculate_noise_count(img_obj, w, h):
    """
    计算邻域非白色的个数
    Args:
        img_obj: img obj
        w: width
        h: height
    Returns:
        count (int)
    """
    count = 0
    width, height, s = img_obj.shape
    for _w_ in [w - 1, w, w + 1]:
        for _h_ in [h - 1, h, h + 1]:
            if _w_ > width - 1:
                continue
            if _h_ > height - 1:
                continue
            if _w_ == w and _h_ == h:
                continue
            if (img_obj[_w_, _h_, 0] < 233) or (img_obj[_w_, _h_, 1] <
                                                233) or (img_obj[_w_, _h_, 2] <
                                                         233):
                count += 1
    return count


# k邻域降噪
def operate_img(img, k):
    w, h, s = img.shape
    # 从高度开始遍历
    for _w in range(w):
        # 遍历宽度
        for _h in range(h):
            if _h != 0 and _w != 0 and _w < w - 1 and _h < h - 1:
                if calculate_noise_count(img, _w, _h) < k:
                    img.itemset((_w, _h, 0), 255)
                    img.itemset((_w, _h, 1), 255)
                    img.itemset((_w, _h, 2), 255)

    return img


def around_white(img):
    w, h, s = img.shape
    for _w in range(w):
        for _h in range(h):
            if (_w <= 5) or (_h <= 5) or (_w >= w - 5) or (_h >= h - 5):
                img.itemset((_w, _h, 0), 255)
                img.itemset((_w, _h, 1), 255)
                img.itemset((_w, _h, 2), 255)
    return img


# 邻域非同色降噪
def noise_unsome_piexl(img):
    '''
        查找像素点上下左右相邻点的颜色，如果是非白色的非像素点颜色，则填充为白色
    :param img:
    :return:
    '''
    # print(img.shape)
    w, h, s = img.shape
    for _w in range(w):
        for _h in range(h):
            if _h != 0 and _w != 0 and _w < w - 1 and _h < h - 1:  # 剔除顶点、底点
                center_color = img[_w, _h]  # 当前坐标颜色
                # print(center_color)
                top_color = img[_w, _h + 1]
                bottom_color = img[_w, _h - 1]
                left_color = img[_w - 1, _h]
                right_color = img[_w + 1, _h]
                cnt = 0
                if all(top_color == center_color):
                    cnt += 1
                if all(bottom_color == center_color):
                    cnt += 1
                if all(left_color == center_color):
                    cnt += 1
                if all(right_color == center_color):
                    cnt += 1
                if cnt < 1:
                    img.itemset((_w, _h, 0), 255)
                    img.itemset((_w, _h, 1), 255)
                    img.itemset((_w, _h, 2), 255)
    return img


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "train"

# 加载模型标签（以便我们可以将模型预测转换为实际字母）
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# 加载训练好的神经网络
model = load_model(MODEL_FILENAME)

# 随机获取一些验证码图像进行测试。
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(20, ), replace=False)
# captcha_image_files = ['1/1.jpg']

count1 = 0
total = 0
# 在图像路径上循环
for image_file in captcha_image_files:
    # 加载图像并将其转换为灰度
    image = cv2.imread(image_file)
    # plt.subplot(121), plt.imshow(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 在图像周围添加一些额外的填充
    # image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # 设置图像阈值（将其转换为纯黑白）
    thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)[1]
    for i in range(2):
        thresh = operate_img(thresh, 4)
    img = around_white(thresh)
    img = noise_unsome_piexl(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # plt.subplot(122), plt.imshow(thresh)
    # plt.show()
    # 找到图像的轮廓（连续的像素点）
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    # Hack与不同OpenCV版本的兼容性
    # contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = contours[1] if imutils.is_cv3() else contours[0]

    letter_image_regions = []

    # 现在我们可以遍历四个轮廓中的每一个并提取每个轮廓中的字母
    for contour in contours:
        # 获取包含轮廓的矩形
        (x, y, w, h) = cv2.boundingRect(contour)
        if w + h < 40 or h < 10 or w < 5:
            continue
        # 比较轮廓的宽度和高度以检测连成一块的字母
        '''if w / h > 0.8 and len(letter_image_regions) < 4:
            # 这个轮廓太宽了，不可能是一个字母！把它分成两个字母区域
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))'''
        # 这是一封普通的字母
        letter_image_regions.append((x, y, w, h))

    # 如果我们在验证码中发现多于或少于4个字母，我们的字母提取工作不正常。跳过图像而不是保存错误的训练数据！
        if len(letter_image_regions) != 4:
            size_regions = len(letter_image_regions)
            if size_regions < 4:
                if size_regions == 0:
                    continue
                elif size_regions == 1:
                    (x, y, w, h) = letter_image_regions.pop()
                    half_width = int(w / 4)
                    letter_image_regions.append((x, y, half_width, h))
                    letter_image_regions.append((x + half_width, y, half_width, h))
                    letter_image_regions.append(
                        (x + 2 * half_width, y, half_width, h))
                    letter_image_regions.append(
                        (x + 3 * half_width, y, half_width, h))
                elif size_regions == 2:
                    letter_image_regions = sorted(letter_image_regions, key=lambda w: w[0])
                    (x1, y1, w1, h1) = letter_image_regions.pop()
                    (x2, y2, w2, h2) = letter_image_regions.pop()
                    if w1 > 2 * w2:
                        half_width = int(w1 / 3)
                        letter_image_regions.append((x2, y2, w2, h2))
                        letter_image_regions.append((x1, y1, half_width, h1))
                        letter_image_regions.append(
                            (x1 + half_width, y1, half_width, h1))
                        letter_image_regions.append(
                            (x1 + 2 * half_width, y1, half_width, h1))
                    else:
                        half_width = int(w1 / 2)
                        letter_image_regions.append((x1, y1, half_width, h1))
                        letter_image_regions.append(
                            (x1 + half_width, y1, half_width, h1))
                        half_width = int(w2 / 2)
                        letter_image_regions.append((x2, y2, half_width, h2))
                        letter_image_regions.append(
                            (x2 + half_width, y2, half_width, h2))
                elif size_regions == 3:
                    letter_image_regions = sorted(letter_image_regions, key=lambda w: w[0])
                    (x, y, w, h) = letter_image_regions.pop()
                    half_width = int(w / 2)
                    letter_image_regions.append((x, y, half_width, h))
                    letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions = sorted(letter_image_regions, reverse=True, key=lambda w: w[0])
                for i in range(size_regions - 4):
                    (x, y, w, h) = letter_image_regions.pop()
                for i in range(3):
                    (x1, y1, w1, h1) = letter_image_regions.pop()
                    if w1 / w < 1.5:
                        break
                    else:
                        x = x1, y = y1, w = w1, h = h1
                letter_image_regions.append((x1, y1, w1, h1))
                size_regions = len(letter_image_regions)
                if size_regions < 4:
                    if size_regions == 0:
                        continue
                    elif size_regions == 1:
                        (x, y, w, h) = letter_image_regions.pop()
                        half_width = int(w / 4)
                        letter_image_regions.append((x, y, half_width, h))
                        letter_image_regions.append((x + half_width, y, half_width, h))
                        letter_image_regions.append(
                            (x + 2 * half_width, y, half_width, h))
                        letter_image_regions.append(
                            (x + 3 * half_width, y, half_width, h))
                    elif size_regions == 2:
                        letter_image_regions = sorted(letter_image_regions, key=lambda w: w[0])
                        (x1, y1, w1, h1) = letter_image_regions.pop()
                        (x2, y2, w2, h2) = letter_image_regions.pop()
                        if w1 > 2 * w2:
                            half_width = int(w1 / 3)
                            letter_image_regions.append((x2, y2, w2, h2))
                            letter_image_regions.append((x1, y1, half_width, h1))
                            letter_image_regions.append(
                                (x1 + half_width, y1, half_width, h1))
                            letter_image_regions.append(
                                (x1 + 2 * half_width, y1, half_width, h1))
                        else:
                            half_width = int(w1 / 2)
                            letter_image_regions.append((x1, y1, half_width, h1))
                            letter_image_regions.append(
                                (x1 + half_width, y1, half_width, h1))
                            half_width = int(w2 / 2)
                            letter_image_regions.append((x2, y2, half_width, h2))
                            letter_image_regions.append(
                                (x2 + half_width, y2, half_width, h2))
                    elif size_regions == 3:
                        letter_image_regions = sorted(letter_image_regions, key=lambda w: w[0])
                        (x, y, w, h) = letter_image_regions.pop()
                        half_width = int(w / 2)
                        letter_image_regions.append((x, y, half_width, h))
                        letter_image_regions.append((x + half_width, y, half_width, h))
    total = total + 4
    # 根据x坐标对检测到的字母图像进行排序，以确保从左到右进行处理，以便将图像与字母匹配
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # 创建一个输出图像和一个列表来保存我们预测的字母
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = cv2.merge([output] * 3)
    predictions = []

    # 循环
    for letter_bounding_box in letter_image_regions:
        # 获取图像中字母的坐标
        x, y, w, h = letter_bounding_box
        if y > 2:
            y = y - 2
        else:
            y = 0
        if x > 2:
            x = x - 2
        else:
            x = 0
        # 从边缘有2个像素边距的原始图像中提取字母
        letter_image = gray[y:y + h + 4, x:x + w + 4]
        if letter_image.size == 0:
            continue

        # 将字母图像的大小重新调整为20x20像素，以匹配训练数据
        letter_image = resize_to_fit(letter_image, 20, 20)

        # 将单个图像转换为4d图像列表以适配Keras
        letter_image = np.expand_dims(letter_image, axis=2)
        letter_image = np.expand_dims(letter_image, axis=0)

        # 让神经网络做出预测
        prediction = model.predict(letter_image)

        # 将一个独热编码预测转换回正常字母
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # 在输出图像上绘制预测
        cv2.rectangle(output, (x, y), (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # 打印验证码文本
    count2 = 0
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))
    captcha_correct_text = os.path.splitext(image_file)[0][-4:]
    print("CORRECT text is: {}".format(captcha_correct_text))
    for i in range(4):
        if captcha_correct_text[i] == captcha_text[i]:
            count1 += 1
            count2 += 1
    print("Correct Number:{}".format(count2))

print("Total Correct Numbers:{}".format(count1))
print("Accuracy:{}".format(count1/total))
# 显示带批注的图像
# cv2.imshow("Output", output)
# cv2.waitKey()
