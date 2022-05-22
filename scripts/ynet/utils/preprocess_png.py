import cv2
import numpy as np

def get_segmentation_from_png(file,params,division_factor):
    image = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    image[image < 230] = 0
    image[image >= 230] = 255
    image = cv2.GaussianBlur(image, (7,7), 3)
    image[image < 128] = 0
    image[image >= 128] = 2*image[image >= 128] - 255

    # Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
    image = cv2.resize(image, (0, 0), fx=params['resize'], fy=params['resize'], interpolation=cv2.INTER_NEAREST)
    H, W = image.shape
    H_new = int(np.ceil(H / division_factor) * division_factor)
    W_new = int(np.ceil(W / division_factor) * division_factor)
    image = cv2.copyMakeBorder(image, 0, H_new - H, 0, W_new - W, cv2.BORDER_CONSTANT)

    H, W = image.shape
    rt = np.zeros((1, 6, H, W))
    rt[0, 1, :, :] = image / 255.
    rt[0, 4, :, :] = 1 - image / 255.

    return rt

# image = get_segmentation_from_png("../../../config/free_space_301_1f.png", {'resize':0.3}, 32)
# cv2.imshow("image", image[0,4,:,:])
# cv2.waitKey(1000000)