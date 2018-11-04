from PIL import Image, ImageChops, ImageOps, ImageFilter

img = Image.open('./img/milk.jpg')
qant_img = img.quantize(32, kmeans=True).convert("RGB")

t = qant_img
t = t.filter(ImageFilter.ModeFilter(10))
t = t.filter(ImageFilter.GaussianBlur(1))
#t = t.filter(ImageFilter.ModeFilter(12))
#t = t.filter(ImageFilter.GaussianBlur(1))
color_img = t.convert("RGB")
'''
gray = img.convert("L")
gray2 = gray.filter(ImageFilter.MaxFilter(5))
line_inv = ImageChops.difference(gray, gray2)
line_img = ImageOps.invert(line_inv).convert("RGB")

ImageChops.multiply(color_img, line_img)
'''

color_img.save('./img/Local_equalization/oilpaint.jpg')
