import cv2
import numpy as np

img = cv2.imread("src/test/wheat-rust-filter-test.jpg")

# cv2.imshow("1",img)
# cv2.waitKey(0)
filter1 = np.array([
                [-1, 0, 0],
                [0, 0, 0],
                [0, 0, 1]])

res = cv2.filter2D(img, -1, filter1)

cv2.imshow("2", res)
cv2.imwrite("src/test/2.jpg", res)
cv2.waitKey(0)
