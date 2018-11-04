import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./img/milk.jpg', 0)
lap = cv2.Laplacian(img, cv2.CV_32F,ksize=1)
lap5 = cv2.Laplacian(img, cv2.CV_32F,ksize=3)

plt.subplot(1,2,1),plt.imshow(lap,cmap = 'gray')
plt.title('Laplacian K=4'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(lap5,cmap = 'gray')
plt.title('Laplacian k=8'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.imwrite('./img/laplacian/output1.jpg', lap)
cv2.imwrite('./img/laplacian/output2.jpg', lap5)
