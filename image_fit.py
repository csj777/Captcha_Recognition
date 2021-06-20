import imutils
import cv2


def resize_to_fit(image, width, height):
    """
    调整图像大小以适应给定大小的辅助函数
    ：param image:要调整大小的图像
    ：param width：所需的像素宽度
    ：param height：所需高度（像素）
    ：return：调整大小的图像
    """

    # 获取图像的维度，然后初始化填充值
    (h, w) = image.shape[:2]

    # 如果宽度大于高度，则沿宽度调整大小
    if w > h:
        image = imutils.resize(image, width=width)

    # 否则，高度大于宽度，因此沿高度调整大小
    else:
        image = imutils.resize(image, height=height)

    # 确定宽度和高度的填充值以获得目标尺寸
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # 填充图像，然后再应用一个大小调整以处理任何舍入问题
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # 返回预处理图像
    return image
