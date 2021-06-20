import os
import os.path
import cv2
import glob
import imutils

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


CAPTCHA_IMAGE_FOLDER = "train"
OUTPUT_FOLDER = "extracted_letter_images"

# 得到所有图片
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# 循环访问所有图片
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(captcha_image_files)))

    # 获取基本文件名作为文本
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # 加载图像并将其转换为灰度
    image = cv2.imread(captcha_image_file)
    thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)[1]
    for i in range(3):
        thresh = operate_img(thresh, 4)
    img = around_white(thresh)
    img = noise_unsome_piexl(img)
    # plt.subplot(121), plt.imshow(image)
    # plt.subplot(122), plt.imshow(img)
    # plt.show()
    # 找到图像的轮廓（连续的像素点
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # plt.subplot(144), plt.imshow(thresh)
    # plt.show()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    # Hack与不同OpenCV版本的兼容性
    # contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = contours[1] if imutils.is_cv3() else contours[0]

    letter_image_regions = []

    # 遍历四个轮廓并提取字母
    for contour in contours:
        # 获取包含轮廓的矩形
        (x, y, w, h) = cv2.boundingRect(contour)
        # 如果轮廓的长宽较小，那么应该是噪点
        if w + h < 40 or w < 10 or h < 20:
            continue
        # 比较轮廓的宽度和高度以检测连成一块的数字
        '''if w / h > 0.8 and len(letter_image_regions) < 4:
            # 轮廓太宽了，可能是多个字母合并的，进行分割
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))'''
        # 这是一个普通的数字
        letter_image_regions.append((x, y, w, h))

    # 如果我们在验证码中发现多于或少于4个字母，进行处理
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
                letter_image_regions = sorted(letter_image_regions,
                                              key=lambda w: w[0])
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
                letter_image_regions = sorted(letter_image_regions,
                                              key=lambda w: w[0])
                (x, y, w, h) = letter_image_regions.pop()
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions = sorted(letter_image_regions,
                                          reverse=True,
                                          key=lambda w: w[0])
            for i in range(size_regions - 4):
                (x, y, w, h) = letter_image_regions.pop()
            for i in range(3):
                (x1, y1, w1, h1) = letter_image_regions.pop()
                if w1 / w < 1.5:
                    break
                else:
                    x = x1, y = y1, w = w1, h = h1
            letter_image_regions.append(x1, y1, w1, h1)
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
    # 根据x坐标对检测到的字母图像进行排序，以确保从左到右进行处理，以便将图像与字母匹配
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # 将每个字母保存为单个图像
    for letter_bounding_box, letter_text in zip(letter_image_regions,
                                                captcha_correct_text):
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

        # 获取保存图像的文件夹
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # 如果输出目录不存在，则创建它
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 将字母图像写入文件
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # 增加当前的计数
        counts[letter_text] = count + 1
