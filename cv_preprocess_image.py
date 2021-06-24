import cv2
import numpy as np
import os
from skimage import morphology
from matplotlib import pyplot as plt

for index, path in enumerate(os.listdir('./test')):
    img = cv2.imread(os.path.join('./test', path))
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(src=img1, thresh=220, maxval=255, type=cv2.THRESH_BINARY)
    median = cv2.medianBlur(threshold, 5)
    kernel = np.ones((2, 2), dtype=np.uint8)
    dilation = cv2.dilate(src=median, kernel=kernel, iterations=1)
    binarized = np.where(dilation > 0.1, 1, 0)
    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=50, connectivity=1).astype(int)
    mask_x, mask_y = np.where(processed == 0)
    dilation[mask_x, mask_y] = 0

    plt.subplot(3, 4, index+1)
    plt.imshow(dilation, cmap='gray')
    # 将原图中的目标区域覆盖到mask上
    img_fg = cv2.bitwise_and(img, img, mask=dilation)
    # plt.imshow(img_fg)
plt.show()
