import os
import os.path
import cv2
import glob
import imutils
# import matplotlib.pyplot as plt

CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


# 得到所有图片
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# 循环访问所有图片
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # 获取基本文件名作为文本
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # 加载图像并将其转换为灰度
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 在图像周围添加一些额外的填充
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    # 设置图像阈值（将其转换为纯黑白）
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # plt.subplot(144), plt.imshow(thresh)
    # plt.show()
    # 找到图像的轮廓（连续的像素点
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack与不同OpenCV版本的兼容性
    # contours = contours[0] if imutils.is_cv2() else contours[1]
    contours = contours[1] if imutils.is_cv3() else contours[0]

    letter_image_regions = []

    # 现在我们可以遍历四个轮廓并提取字母
    # 在每一个里面
    for contour in contours:
        # 获取包含轮廓的矩形
        (x, y, w, h) = cv2.boundingRect(contour)

        # 比较轮廓的宽度和高度以检测连成一块的数字
        if w / h > 1.25:
            # 这个轮廓太宽了，不可能是一个字母！
            # 把它分成两个字母区域！
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # 这是一个普通的数字
            letter_image_regions.append((x, y, w, h))

    # 如果我们在验证码中发现多于或少于4个字母，我们的字母提取失败，下一个
    if len(letter_image_regions) != 4:
        continue

    # 根据x坐标对检测到的字母图像进行排序，以确保从左到右进行处理，以便将图像与字母匹配
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # 将每个字母保存为单个图像
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # 获取图像中字母的坐标
        x, y, w, h = letter_bounding_box

        # 从边缘有2个像素边距的原始图像中提取字母
        letter_image = gray[y:y + h + 2, x:x + w + 2]

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
