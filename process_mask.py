#!/usr/bin/python3

#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from PIL import Image
import scipy.ndimage
import scipy.misc
import floodfill

class Img:
    def __init__(self, data, path):
        self.data = data
        self.path = path

    def store(self):
        scipy.misc.imsave(self.path, self.data)

    def show(self):
        Image.open(self.path).show()

def img_from_file(path):
    data = np.array(Image.open(path))
    return Img(data, path)

def to_binary(img, lower, upper):
    return (lower < img) & (img < upper)

def insert_postfix(path, postfix):
    index = path.rfind('.')
    return ''.join([path[:index], '_', postfix, path[index:]])

def make_binary(img):
    bin_data = 1.0 * to_binary(img.data, 20000, 70000)
    bin_path = insert_postfix(img.path, 'bin')
    bin_img = Img(bin_data, bin_path)
    bin_img.store()
    return bin_img

def edge_fill(img):
    filled_data = floodfill.from_edges(img.data)
    filled_path = insert_postfix(img.path, 'edge-filled')
    filled_img = Img(filled_data, filled_path)
    filled_img.store()
    return filled_img

def point_fill(img, points):
    filled_data = floodfill.from_points(img.data, points)
    filled_path = insert_postfix(img.path, 'points-filled')
    filled_img = Img(filled_data, filled_path)
    filled_img.store()
    return filled_img

def fill_binary(img):
    filled_data = 1.0 * scipy.ndimage.morphology.binary_fill_holes(img.data)
    filled_path = insert_postfix(img.path, "bin-filled")
    filled_img = Img(filled_data, filled_path)
    filled_img.store()
    return filled_img

def main():
    os.putenv("DISPLAY", ":0.0")
    data_dir = os.path.join(os.path.curdir, 'data')
    mask_filename = '*_mask_all.tif'
    for filepath in glob.glob(os.path.join(data_dir, mask_filename)):
        img = img_from_file(filepath)
        img_bin = make_binary(img)

if __name__ == "__main__":
    main()
