import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from PIL import Image


def get_image():
    delta = 0.25
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X**2 - Y**2)
    Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
    Z = (Z1 - Z2)
    return Z


def do_plot(ax, Z, transform):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=[-2, 4, -3, 2], clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)

#アフィン変換で回転移動処理をした画像を保存する.
def Rotation_mv_matrix(img,degree,title):
    h, w = img.shape[:2]
    size = (w, h)
    # 回転角の指定
    angle = degree
    angle_rad = angle/180.0*np.pi
    # 回転後の画像サイズを計算
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)
    # 元画像の中心を軸に回転する
    center = (w/2, h/2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # 平行移動を加える (rotation + translation)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2
    #img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)
    #cv2.imwrite(title,img_rot)
    return img_rot,size_rot

def crop_center(pil_img, crop_width, crop_height,size_rot):
    img_width  = size_rot[0]
    img_height = size_rot[1]
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

#画像の読み込み
img = np.array(Image.open('./img/lena.jpg'))
img = np.flip(img,axis=0)
#画像の表示
plt.imshow(img)
h, w = img.shape[:2]

#求めたい任意の直線ax+by+c=0を定義する.
a = -39
b = 56
c = -a*(w/2) - b*(h/2)
cos = b / np.sqrt(a**2 + b**2)
sin = a / np.sqrt(a**2 + b**2)
degree =  math.degrees(math.asin(sin))
x = np.linspace(0,w,w+1)
y = np.linspace(0,h,h+1)
y = -a*x/b-c/b
print(c)
print(degree)
ax = plt.subplot()
ax.patch.set_facecolor('black')
plt.ylim(0,h)
plt.xlim(0,w)
plt.plot(x,y,"r-")
plt.savefig("./img/affine/result3.jpg")
plt.show()

img2,size_rot = Rotation_mv_matrix(img,-degree,0)
img3 = cv2.flip(img2,0)
img4,size_rot = Rotation_mv_matrix(img3,degree,0)
Image.fromarray(img4).save('./img/affine/output.jpg')
im = Image.open('./img/affine/output.jpg')
im_new = crop_center(im, w, h, size_rot)
#print(size_rot)
plt.imshow(im_new)
x = np.linspace(0,w,w+1)
y = np.linspace(0,h,h+1)
y = -a*x/b-c/b
plt.ylim(0,h)
plt.xlim(0,w)
plt.plot(x,y,"r-")
ax = plt.subplot()
ax.patch.set_facecolor('black')
plt.savefig("./img/affine/result4.jpg")
plt.show()

# prepare image and figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
Z = get_image()

# image rotation
do_plot(ax1, img, mtransforms.Affine2D().rotate_deg(30))

# image skew
do_plot(ax2, Z, mtransforms.Affine2D().skew_deg(30, 15))

# scale and reflection
do_plot(ax3, Z, mtransforms.Affine2D().scale(-1, .5))

# everything and a translation
do_plot(ax4, Z, mtransforms.Affine2D().
        rotate_deg(30).skew_deg(30, 15).scale(-1, .5).translate(.5, -1))

plt.show()
