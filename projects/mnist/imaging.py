import math
from PIL import Image

def to_picture(array):
    img = Image.new("RGB",(28,28))
    pixels = img.load()
    for i,p in enumerate(array):
        w = i % 28
        h = i / 28
        tp = int(p * 255)
        pixels[w,h] = (tp,tp,tp)
    return img

def combine(imgs):
    length = len(imgs)
    start = int(math.floor(math.sqrt(length)))
    while length%start != 0:
        start -= 1

    h = start
    w = length/start

    img = Image.new("RGB",(w*28,h*28))
    for i in range(w):
        for j in range(h):
            o = i + j * w
            img.paste(imgs[o],(i*28,j*28))

    return img

def combine_mnist(sets):
    return combine(map(to_picture,sets))
