# Description: This script uses the average pixel color of an image to classify skin color on the Fitzpatrick scale. 
# We must use the Von Luschan scale instead of the Fitzpatrick scale because we have rgb values for the Von Luschan scale.

import os
import re
import numpy as np
import scipy.spatial.distance as distance
import PIL.Image as Image
import sys

def get_image_rgb_array(image_path):
    print("get_image_rgb_array")
    image = Image.open(image_path)
    image_rgb_array = np.array(image.convert('RGB'))

    return image_rgb_array

def get_median_pixel_color(image_rgb_array):
    print("get_median_pixel_color")
    # Get the median pixel color
    median_pixel_color = np.median(image_rgb_array, axis=(0, 1))
    print("median_pixel_color: ", median_pixel_color)

    return median_pixel_color

def get_array_of_colors_between_two_rgb_values(rgb1, rgb2):
    print("get_array_of_colors_between_two_rgb_values")
    from PIL import Image
    import numpy as np

    # Create an array of colors between rgb1 and rgb2
    color_array = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        for j in range(100):
            color_array[i, j, 0] = rgb1[0] + (rgb2[0] - rgb1[0]) * i / 100
            color_array[i, j, 1] = rgb1[1] + (rgb2[1] - rgb1[1]) * i / 100
            color_array[i, j, 2] = rgb1[2] + (rgb2[2] - rgb1[2]) * i / 100

    return color_array

def get_closest_color_index(color, color_array):
    print("get_closest_color_index")
    # Get the index of the closest color in color_array dictionary. the dictionary contains vectors of colors in (r, g, b) format.
    # the color parameter is an      of 3 values in [r, g, b] format.
    closest_color_index = 0
    #search the dictionary for the object that contains an array of colors that is closest to the color parameter
    least_distance = distance.euclidean([0, 0, 0], [255, 255, 255])
    for key, value in color_array.items():
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                dictionary_color = value[i, j]
                color_distance = distance.euclidean(color, dictionary_color)
                if color_distance < least_distance:
                    least_distance = color_distance
                    closest_color_index = key
    
    print("closest_color_index: ", closest_color_index, "least_color_distance: ", least_distance)

    return closest_color_index

def convert_von_luschan_scale_to_fitzpatrick_scale(von_luschan_scale_index):
    print("convert_von_luschan_scale_to_fitzpatrick_scale")
    # Define the Fitzpatrick scale
    fitzpatrick_scale = {
        'I': (1, 5),
        'II': (6, 10),
        'III': (11, 15),
        'IV': (16, 21),
        'V': (22, 28),
        'VI': (29, 36)
    }
    # Convert the Von Luschan scale to the Fitzpatrick scale
    for key, value in fitzpatrick_scale.items():
        if von_luschan_scale_index in range(value[0], value[1] + 1):
            return key  
        
def remove_all_non_luschan_pixels(image_rgb_array, color_array, luschan_scale_index):   
    print("remove_all_non_luschan_pixels")
    image_rgb_array_new = np.zeros_like(image_rgb_array, dtype=np.uint8)
    # Remove all pixels that are not in any object in the Von Luschan scale dictionary
    for i in range(image_rgb_array.shape[0]):
        for j in range(image_rgb_array.shape[1]):
            if image_rgb_array[i, j] not in color_array[luschan_scale_index]:
                image_rgb_array_new[i, j] = [0, 0, 0]
            else:
                image_rgb_array_new[i, j] = image_rgb_array[i, j]

    return image_rgb_array_new

def get_array_of_color_distance(color_array, color):
    print("get_array_of_color_distance")
    # Get the array of color distances between color_array and color
    color_distance_array = np.zeros_like(color_array, dtype=np.uint8)
    for i in range(color_array.shape[0]):
        for j in range(color_array.shape[1]):
            for k in range(color_array.shape[2]):
                color_distance_array[i, j, k] = color[k] - color_array[i, j, k]
    
    print("average color distance: ", np.average(color_distance_array))
    print("shape of color_distance_array: ", color_distance_array.shape)
    print("color_distance_array_sample: ", color_distance_array[10,10,1])
    print("color_distance_array_sample: ", color_distance_array[10,40,1])
    return color_distance_array

def get_color_range_from_color_and_distance_array(median_color, color_distance_array, include_color_array=False, original_color_array=None, tolerance=10):
    print("get_color_range_from_color_and_distance_array: median_color: ", median_color)
    # Get the color range of color_array that is within distance of color
    color_range = np.zeros_like(color_distance_array, dtype=np.uint8)
    #get items in color_distance_array
    cells_to_process = color_distance_array.shape[0] * color_distance_array.shape[1] * color_distance_array.shape[2]
    for i in range(color_distance_array.shape[0]):
        for j in range(color_distance_array.shape[1]):
            for k in range(color_distance_array.shape[2]):
                cells_to_process -= 1
                if(include_color_array is not None and original_color_array is not None):
                    new_color = median_color[k] + color_distance_array[i, j, k]
                    #get closest color in include_color_array
                    closest_color_index = get_closest_color_index(new_color, include_color_array)
                    #get the distance between the new color and the closest color in include_color_array
                    closest_color = include_color_array[closest_color_index][0]
                    #if the distance is less than tolerance, set the color to the closest color in include_color_array
                    if distance.euclidean(new_color, closest_color) < tolerance:
                        color_range[i, j, k] = closest_color
                else:
                    color_range[i, j, k] = median_color[k] + color_distance_array[i, j, k]
                

    return color_range

def add_two_rgb_arrays(rgb_array1, rgb_array2):
    print("add_two_rgb_arrays")
    # Add two rgb arrays
    rgb_array_sum = np.zeros_like(rgb_array1, dtype=np.uint8)
    for i in range(rgb_array1.shape[0]):
        for j in range(rgb_array1.shape[1]):
            for k in range(rgb_array1.shape[2]):
                rgb_array_sum[i, j, k] = rgb_array1[i, j, k] + rgb_array2[i, j, k]

    return rgb_array_sum
                
def get_max_distance_betwewen_a_color_and_an_array_of_colors(color, color_array):
    print("get_max_distance_betwewen_a_color_and_an_array_of_colors")
    # Get the max distance between a color and an array of colors
    max_distance = 0
    for i in range(color_array.shape[0]):
        for j in range(color_array.shape[1]):
            color_distance = distance.euclidean(color, color_array[i, j])
            if color_distance > max_distance:
                max_distance = color_distance

    return max_distance

def scale_rgb_array_to_max_distance(rgb_array, max_distance):
    print("scale_rgb_array_to_max_distance", max_distance)
    # Scale rgb_array to max color distance
    rgb_array_scaled = np.zeros_like(rgb_array, dtype=np.uint8)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            for k in range(rgb_array.shape[2]):
                rgb_array_scaled[i, j, k] = rgb_array[i, j, k] * 255 / max_distance

    return rgb_array_scaled

def scale_rgb_array_count_to_another(rgb_array1, rgb_array2):
    print("scale_rgb_array_count_to_another")
    # Scale rgb_array1's indecies to rgb_array2's indecies
    rgb_array_scaled = np.zeros_like(rgb_array2, dtype=np.uint8)
    scale_x = rgb_array1.shape[0] / rgb_array2.shape[0]
    scale_y = rgb_array1.shape[1] / rgb_array2.shape[1]
    for i in range(rgb_array2.shape[0]):
        for j in range(rgb_array2.shape[1]):
            #get the nearest whole number index in rgb_array1
            index_x = int(i * scale_x)
            index_y = int(j * scale_y)
            rgb_array_scaled[i, j] = rgb_array1[index_x, index_y]

    return rgb_array_scaled
    
def multiply_rgb_array_by_another(rgb_array1, rgb_array2):
    print("multiply_rgb_array_by_another")
    # Multiply two rgb arrays
    rgb_array_product = np.zeros_like(rgb_array1, dtype=np.uint8)
    for i in range(rgb_array1.shape[0]):
        for j in range(rgb_array1.shape[1]):
            for k in range(rgb_array1.shape[2]):
                rgb_array_product[i, j, k] = rgb_array1[i, j, k] * rgb_array2[i, j, k]

    return rgb_array_product

def map_rgb_array_to_closest_color(rgb_array, color_array):
    print("map_rgb_array_to_closest_color")
    # Map rgb_array to the closest color in color_array
    rgb_array_new = np.zeros_like(rgb_array, dtype=np.uint8)
    #check if the color_array is a list
    if isinstance(color_array, list):
        for i in range(rgb_array.shape[0]):
            for j in range(rgb_array.shape[1]):
                closest_color_index = get_closest_color_index(rgb_array[i, j], color_array)
                rgb_array_new[i, j] = color_array[closest_color_index]
    elif isinstance(color_array, np.ndarray):
        for i in range(rgb_array.shape[0]):
            for j in range(rgb_array.shape[1]):
                closest_color_index = get_closest_color_index(rgb_array[i, j], color_array)
                rgb_array_new[i, j] = color_array[closest_color_index]
    
    
    return rgb_array_new


def save_rgb_array_as_image(rgb_array, output_path):
    print("save_rgb_array_as_image")
    from PIL import Image
    import numpy as np

    # Output the image
    image = Image.fromarray(rgb_array)
    image.save(output_path)


def define_von_luschan_scale():
    print("define_von_luschan_scale")
    # Define the Von Luschan scale
    rgb_values = [
        (244,242,245),
        (236,235,233),
        (250,249,247),
        (253,251,230),
        (253,246,230),
        (254,247,229),
        (250,240,239),
        (243,234,229),
        (244,241,234),
        (251,252,244),
        (252,248,237),
        (254,246,225),
        (255,249,225),
        (255,249,225),
        (241,231,195),
        (239,226,173),
        (224,210,147),
        (242,226,151),
        (235,214,159),
        (235,217,133),
        (227,196,103),
        (225,193,106),
        (223,193,123),
        (222,184,119),
        (199,164,100),
        (188,151,98),
        (156,107,67),
        (142,88,62),
        (121,77,48),
        (100,49,22),
        (101,48,32),
        (96,49,33),
        (87,50,41),
        (64,32,21),
        (49,36,41),
        (27,28,46)
    ]
    # get the rgb values for the colors between each pair of rgb values in the Von Luschan scale
    von_luschan_scale = {}
    for i in range(len(rgb_values) - 1):
        von_luschan_scale[i] = get_array_of_colors_between_two_rgb_values(rgb_values[i], rgb_values[i + 1])
    
    return von_luschan_scale

def flatten_van_luschan_scale(von_luschan_scale):
    print("flatten_van_luschan_scale")
    # Flatten the Von Luschan scale
    rgb_array_flat = np.zeros((100, 100, 3), dtype=np.uint8)
    for key, value in von_luschan_scale.items():
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                rgb_array_flat[i, j] = value[i, j]

    return rgb_array_flat
def get_color_array_for_fitzpatrick_scale(fitzpatrick_scale_index):
    print("get_color_array_for_fitzpatrick_scale")
    # Define the Fitzpatrick scale
    fitzpatrick_scale = {
        'I': (1, 5),
        'II': (6, 10),
        'III': (11, 15),
        'IV': (16, 21),
        'V': (22, 28),
        'VI': (29, 36)
    }
    # Define the Von Luschan scale
    von_luschan_scale = define_von_luschan_scale()
    # Get the color array for the Fitzpatrick scale
    color_array = {}
    for key, value in fitzpatrick_scale.items():
        if key == fitzpatrick_scale_index:
            for i in range(value[0], value[1] + 1):
                color_array[i] = von_luschan_scale[i]
    
    return color_array


# main
# Define the Von Luschan scale
von_luschan_scale = define_von_luschan_scale()
#flatten the von luschan scale
von_luschan_scale_flat = flatten_van_luschan_scale(von_luschan_scale)
# open the image from argv[1]
image_rgb_array = get_image_rgb_array(sys.argv[1])
print("image_rgb_array.shape: ", image_rgb_array.shape)
#get closest color index
median_pixel_color = get_median_pixel_color(image_rgb_array)
closest_color_index = get_closest_color_index(median_pixel_color, von_luschan_scale)
print("von_luschan_scale_index: ", closest_color_index)
#convert the closest color index to the Fitzpatrick scale
fitzpatrick_scale_index = convert_von_luschan_scale_to_fitzpatrick_scale(closest_color_index)
print("fitzpatrick_scale_index: ", fitzpatrick_scale_index)
#remve all non luschan pixels
non_luschan = remove_all_non_luschan_pixels(image_rgb_array, von_luschan_scale, closest_color_index)
#save the image
save_rgb_array_as_image(non_luschan, "non_luschan_pixels_removed.png")
#get the distance map of the image
color_distance_array = get_array_of_color_distance(image_rgb_array, median_pixel_color)
#save the image
save_rgb_array_as_image(color_distance_array, "color_distance_array.png")
#rebuild the image with the median pixel color 100, 100, 100
new_median_color = [100, 100, 100]
color_range = get_color_range_from_color_and_distance_array(new_median_color, color_distance_array, von_luschan_scale[closest_color_index], image_rgb_array, 10)
#save the image
save_rgb_array_as_image(color_range, "color_range_replace.png")