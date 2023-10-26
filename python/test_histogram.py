"""
２画像のヒストグラム比較による類似度の算出
"""
import cv2, os
# from opencv_japanese import imread

# dir_path =  os.path.dir_path(__file__)
dir_path = r"C:\Users\isapo\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\examples\person"

image1 = cv2.imread(dir_path + '\haru_mae.png')
image2 = cv2.imread(dir_path + '\haru_yoko.png')
image3 = cv2.imread(dir_path + '\ichiro.JPG')
image4 = cv2.imread(dir_path + '\yumi_mae.png')
image5 = cv2.imread(dir_path + '\yumi_yoko.png')


height = image1.shape[0]
width = image1.shape[1]

img_size = (int(width), int(height))

# 比較するために、同じサイズにリサイズしておく
image1 = cv2.resize(image1, img_size)
image2 = cv2.resize(image2, img_size)
image3 = cv2.resize(image3, img_size)
image4 = cv2.resize(image4, img_size)
image5 = cv2.resize(image5, img_size)


# 画像をヒストグラム化する
image1_hist = cv2.calcHist([image1], [2], None, [256], [0, 256])
image2_hist = cv2.calcHist([image2], [2], None, [256], [0, 256])
image3_hist = cv2.calcHist([image3], [2], None, [256], [0, 256])
image4_hist = cv2.calcHist([image4], [2], None, [256], [0, 256])
image5_hist = cv2.calcHist([image5], [2], None, [256], [0, 256])


# ヒストグラムした画像を比較
print("image1とimage2の類似度：" + str(cv2.compareHist(image1_hist, image2_hist, 0)))
print("image1とimage3の類似度：" + str(cv2.compareHist(image1_hist, image3_hist, 0)))
print("image1とimage4の類似度：" + str(cv2.compareHist(image1_hist, image4_hist, 0)))
print("image1とimage5の類似度：" + str(cv2.compareHist(image1_hist, image5_hist, 0)))

