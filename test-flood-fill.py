#!/usr/bin/python3

import os
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import floodfill
from PIL import Image

#matplotlib.rcParams['figure.figsize'] = (10, 10)

def trial():
    dem_adj = np.array([
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    ])

    outflow_pts = [[0, 5]]
    mask = np.zeros(dem_adj.shape, dtype=np.bool)
    for pt in outflow_pts:
        mask[pt[0],pt[1]] = 1
    #mask[zip(*outflow_pts)] = 1
    print("image:")
    print(dem_adj)
    print("mask:")
    print(mask)

    dem_fill = dem_adj.copy()

    dem_fill = floodfill.from_points(dem_fill, outflow_pts, four_way=False)
    print("result:")
    print(dem_fill)

def test(image_name):
    picture = Image.open(os.path.join(os.path.curdir, 'data', image_name))
    picture.show()
    image = np.array(picture)
    bin_image = 1.0 * to_binary(image, 50, 255)
    scipy.misc.imsave('data/temp.png', bin_image)
    bin_im = Image.open(os.path.join(os.path.curdir, 'data', 'temp.png'))
    bin_im.show()


def main():
    os.putenv("DISPLAY", ":0.0")
    test1()

if __name__ == "__main__":
    main()
