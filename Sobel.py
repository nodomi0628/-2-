import cv2
from matplotlib import pyplot as plt

img = cv2.imread('./img/milk.jpg', 0)
print(img)
sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
sobelx5 = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
sobely5 = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)

plt.subplot(2,2,1),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X k=3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y k=3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx5,cmap = 'gray')
plt.title('Sobel X k=5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely5,cmap = 'gray')
plt.title('Sobel Y k=5'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite('./img/sobel/output1.jpg', sobelx)
cv2.imwrite('./img/sobel/output2.jpg', sobely)
cv2.imwrite('./img/sobel/output3.jpg', sobelx5)
cv2.imwrite('./img/sobel/output4.jpg', sobely5)
