import cv2
import numpy as np

img = cv2.imread("paper/resource/second/rust.jpg")

# cv2.imshow("1",img)
# cv2.waitKey(0)
filter1 = np.array([
                [0, 1, 2],
                [2, 2, 0],
                [0, 1, 2]])
filter2 = np.array([
                [2, 1, 0],
                [0, 2, 2],
                [2, 1, 0]])


it = cv2.filter2D(img, -1, filter1)
iT = cv2.filter2D(img, -1, filter2)

cv2.imshow("1", it)
cv2.imshow("2", iT)
cv2.imwrite("1.jpg", it)
cv2.imwrite("2.jpg", iT)
cv2.waitKey(0)
