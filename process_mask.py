#!/usr/bin/python3

#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
from PIL import Image
import scipy.ndimage
import scipy.misc
import floodfill

sys.setrecursionlimit(10000)
DEBUG = os.getenv("DEBUG")
eight_way = os.getenv("EIGHT_WAY", default=True)

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

def to_binary(img, lower, upper, reverse=False):
    return reverse ^ ((lower < img) & (img < upper))

def insert_postfix(path, postfix):
    index = path.rfind('.')
    return ''.join([path[:index], '_', postfix, path[index:]])

def make_binary(img, lower_threshold=20000, upper_threshold=70000,
        reverse=False):
    bin_data = 1.0 * to_binary(img.data, lower_threshold, upper_threshold,
            reverse)
    if os.getenv("DEBUG") != None:
        print(bin_data)
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

def get_points_with_given_value_from_img_data(data, value):
    points = []
    rows = data.shape[0]
    cols = data.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if data[x,y] == value:
                points.append([x,y])
    return points

def find_nearest_neighbor_with_given_value(data, x, y, value,
        max_look_around_distance):
    for look_around_distance in range(1, max_look_around_distance):
        look_around_window_x_min = max(0,x-look_around_distance)
        look_around_window_x_max = min(data.shape[0]-1,x+look_around_distance)
        look_around_window_y_min = max(0,y-look_around_distance)
        look_around_window_y_max = min(data.shape[1]-1,y+look_around_distance)
        if DEBUG:
            print('look around corners:', [look_around_window_x_min,
                    look_around_window_y_min],
                    [look_around_window_x_max,
                    look_around_window_y_max])
        look_around_window = \
                data[look_around_window_x_min:look_around_window_x_max,
                look_around_window_y_min:look_around_window_y_max]
        if DEBUG:
            print('point coordinates:', '[%d, %d]' % x,y)
            print("data:", data[x,y])
            print("mark:", marks[x,y])
            print("bigger window:\n", bigger_window)
            print('look around window:\n', look_around_window)
        actual_point_relative = get_first_point_with_given_value(look_around_window,
                value)
        if actual_point_relative == None:
            actual_point = actual_point_relative
            continue
        actual_point = [
                max(0,x-look_around_distance) + actual_point_relative[0],
                max(0,y-look_around_distance) + actual_point_relative[1]
                ]
        if DEBUG:
            print('relative coordinates to actual point:',
                    actual_point_relative)
            print('calculated absolute coordinates to actual point:',
                    actual_point)
            print('value (should be 0) at calculated abs coord to actual '+
                    'point:', data[actual_point[0],
                        actual_point[1]])
    return actual_point

def get_first_point_with_given_value(data, value):
    rows = data.shape[0]
    cols = data.shape[1]
    for x in range(0, rows):
        for y in range(0, cols):
            if data[x,y] == value:
                return [x,y]
    return None

def taint_point(data,x,y, eight_way=True):
    ## edge cases: out of bound, so we don't need to worry elsewhere
    if x < 0 or y < 0 or x >= data.shape[0] or y >= data.shape[1]:
        return
    ## recursion: if 1 or 0.5, return
    if data[x,y] == 1 or data[x,y] == 0.5:
        return
    ## only other case is 0, set to 0.5
    data[x,y] = 0.5
    ## call all 8 neighbors
    taint_point(data, x-1,y)
    taint_point(data, x,y+1)
    taint_point(data, x+1,y)
    taint_point(data, x,y-1)
    if eight_way:
        taint_point(data, x-1,y+1)
        taint_point(data, x+1,y+1)
        taint_point(data, x+1,y-1)
        taint_point(data, x-1,y-1)

def remove_marked_regions(data, marks, max_look_around_distance=10):
    if DEBUG:
        print('received following data:\n', data)
        print('... and following mask:\n', marks)
    # iterate through marks
    points = get_points_with_given_value_from_img_data(marks, 0)
    # change value of all regions which are touched by or close to marks from 0 to 0.5
    for point in points:
        print("processing", point)
        # get coordinates of point
        x, y = point[0], point[1]
        # if already 0.5, move on
        if data[x,y] == 0.5:
            continue
        # if 0, start recursion
        elif data[x,y] == 0.0:
            taint_point(data,x,y,eight_way)
        # if 1, look around at all neighbors (as far as look_around_distance),
        # and keep looking
        else:
            while True:
                actual_point = find_nearest_neighbor_with_given_value(data, x,y,
                        marks[x,y], max_look_around_distance)
                if actual_point == None:
                    if DEBUG:
                        print("""No point with value %f found around [%d, %d] within look
                        around distance %d, continuing""" % (marks[x,y], x, y,
                        max_look_around_distance))
                    break
                actual_x = actual_point[0]
                actual_y = actual_point[1]
                taint_point(data,actual_x,actual_y,eight_way)
    if DEBUG:
        print('processed data:\n', data)
    # filter out all the unmarked 0.0 regions and convert back to 0
    unmarked = (data < 0.1).astype(int)
    data += unmarked
    if DEBUG:
        print('removed unmarked 0 regions:\n', data)
    # convert all 0.5 regions back to 0.0 regions
    data = 1 * (0.6 < data)
    if DEBUG:
        print('converted marked regions back to 0:\n', data)
    np.set_printoptions(edgeitems=200)
    if DEBUG:
        print('final result:\n', data)
    return data

def filter_image(img, mask):
    max_look_around_distance_string = os.getenv("MAX_LOOK_AROUND_DIST", "10")
    max_look_around_distance = int(max_look_around_distance_string)
    img_bin = 1.0 * (img.data >= 20)
    mask_bin = 1.0 * (np.min(mask.data, 2) == 255)
    filtered_data = remove_marked_regions(img_bin, mask_bin,
            max_look_around_distance)
    filtered_path = insert_postfix(img.path,
        'filtered_'+max_look_around_distance_string)
    filtered_img = Img(filtered_data, filtered_path)
    filtered_img.store()
    return filtered_img

def fill_binary(img):
    filled_data = 1.0 * scipy.ndimage.morphology.binary_fill_holes(img.data)
    filled_path = insert_postfix(img.path, "bin-filled")
    filled_img = Img(filled_data, filled_path)
    filled_img.store()
    return filled_img

def transform_filtered_annotation_to_bin(data_dir):
    img_glob = '2_*_resized_filtered.tif'
    for img_path in glob.glob(os.path.join(data_dir, img_glob)):
        print(img_path)
        img = img_from_file(img_path)
        print(img.data.shape)
        img_min_data = 1.0 * (np.min(img.data, 2) == 255)
        #img_min_data = 1.0 * (img.data >= 20)
        print(img_min_data)
        #make_binary(img, lower_threshold=100, upper_threshold=260)

def main():
    os.putenv("DISPLAY", ":0.0")
    if DEBUG:
        np.set_printoptions(edgeitems=80)
    data_dir = os.path.join(os.path.curdir, 'data')

    #transform_filtered_annotation_to_bin(data_dir)
    #return

    mask_glob = '2_mask_all_bin_resized.tif'
    for mask_path in glob.glob(os.path.join(data_dir, mask_glob)):
        print('processing', mask_path)
        mask = img_from_file(mask_path)
        annotation_glob = (os.path.basename(mask_path).split('_')[0] +
                '*filtered.tif')
        annotation_path = glob.glob(os.path.join(data_dir, annotation_glob))[0]
        annotation = img_from_file(annotation_path)
        final_mask = filter_image(mask, annotation)

def test():
    data = np.array([
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ])
    mask = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    remove_marked_regions(data,mask)

if __name__ == "__main__":
    main()
    #test()
