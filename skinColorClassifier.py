# Description: This script uses the average pixel color of an image to classify skin color on the Fitzpatrick scale. 
# We must use the Von Luschan scale instead of the Fitzpatrick scale because we have rgb values for the Von Luschan scale.

import os
import re
import numpy as np
import scipy.spatial.distance as distance
import PIL.Image as Image
import cv2
import sys
from scipy.ndimage import binary_dilation

def print_persistent(prompt, message):
    print(prompt, end='', flush=True)
    print("\r", end='', flush=True)
    print(message)
    

def get_image_rgb_array(image_path):
    print("get_image_rgb_array")
    image = Image.open(image_path)
    image_rgb_array = np.array(image.convert('RGB'))
    #print some info about the image
    print("image size: ", image.size)
    print("image mode: ", image.mode)
    print("image format: ", image.format)
    #print the width and height of the image
    width, height = image.size
    print("width: ", width, "height: ", height)
    #get the unique rgb values in the image
    unique_pixels = np.unique(image_rgb_array, axis=0)
    print("unique_pixels: ", unique_pixels.shape)

    return image_rgb_array
def get_median_pixel_color(image_rgb_array, von_luschan_scale=False, green_screen_color=[0, 255, 0]):
    print("get_median_pixel_color")

    # Create boolean masks for green screen and non-von_luschan_scale pixels
    green_screen_mask = np.all(image_rgb_array == green_screen_color, axis=-1)
    non_von_luschan_scale_mask = np.ones_like(green_screen_mask, dtype=bool)

    if von_luschan_scale:
        for _, value in von_luschan_scale.items():
            non_von_luschan_scale_mask &= ~np.isin(image_rgb_array, value).all(axis=-1)

    # Combine masks to filter out unwanted pixels
    valid_pixels_mask = ~(green_screen_mask | non_von_luschan_scale_mask)

    # Apply masks to get valid pixels
    valid_pixels = image_rgb_array[valid_pixels_mask]

    if len(valid_pixels) == 0:
        return [0, 0, 0]  # Return a default color if no valid pixel found

    # Calculate the median pixel color
    median_pixel_color = np.median(valid_pixels, axis=0).astype(np.uint8)
    print("median_pixel_color: ", median_pixel_color)

    return median_pixel_color



def is_color_von_luschon_scale(color, von_luschan_scale):
    # Check if the color is in the Von Luschan scale (it's a dictionary of numpy arrays. The key is the von luschan scale category)
    for key, value in von_luschan_scale.items():
        if color in value:
            return True
        
    return False
    

def get_array_of_colors_between_two_rgb_values(rgb1, rgb2):
    print("get_array_of_colors_between_two_rgb_values")
    min_r, max_r = min(rgb1[0], rgb2[0]), max(rgb1[0], rgb2[0])
    min_g, max_g = min(rgb1[1], rgb2[1]), max(rgb1[1], rgb2[1])
    min_b, max_b = min(rgb1[2], rgb2[2]), max(rgb1[2], rgb2[2])

    # get the amount of cells in the color_range
    cells = (max_r - min_r) * (max_g - min_g) * (max_b - min_b)

    # create a 1d array of zeros with the shape of the number of colors between each pair of rgb values
    color_array = np.zeros((int(cells), 3), dtype=np.uint8)

    # get items in color_array
    for i in range(color_array.shape[0]):
        # get the rgb values for each color in color_array
        color_array[i, 0] = min_r + i // ((max_g - min_g) * (max_b - min_b))
        color_array[i, 1] = min_g + (i // (max_b - min_b)) % (max_g - min_g)
        color_array[i, 2] = min_b + i % (max_b - min_b)

    return color_array


def get_closest_color_inside_array(target_color, color_array):
    # Convert the target_color to a 1D NumPy array
    target_color_array = np.array(target_color)

    # Calculate the Euclidean distances for all colors in color_array
    distances = np.linalg.norm(color_array - target_color_array, axis=1)

    # Find the index of the color with the minimum distance
    closest_index = np.argmin(distances)

    # Return the closest color
    closest_color = color_array[closest_index]

    return closest_color

def get_x_y_index_of_color_or_closest_color_inside_array(color, color_array):
    print("get_x_y_index_of_color_or_closest_color_inside_array")
    # Get the x,y index of the color or closest color inside the array
    closest_color = color_array[0, 0, 0]
    closest_color_distance = 255
    x_index = 0
    y_index = 0
    for i in range(color_array.shape[0]):
        for j in range(color_array.shape[1]):
            for k in range(color_array.shape[2]):
                color_distance = distance.euclidean(color, color_array[i, j])
                if color_distance < closest_color_distance:
                    closest_color_distance = color_distance
                    closest_color = color_array[i, j]
                    x_index = i
                    y_index = j

    print("x_index: ", x_index, "y_index: ", y_index)
    return x_index, y_index


def get_von_lucian_scale_index_from_color(color, von_luschan_scale):
    print("get_von_lucian_scale_index_from_color")
    # Get the Von Luschan scale index from the color
    for key, value in von_luschan_scale.items():
        if np.array_equal(color, value):
            return key


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
        

def compare_all_von_luschan_scales_in_image_array(image_rgb_array, tolerance=100):
    print("compare_all_von_luschan_scales_in_image_array")
    von_luschan_scale = define_von_luschan_scale()
    for i in range(len(von_luschan_scale)):
        image_rgb_array_new = remove_all_pixels_not_found_in_color_array(image_rgb_array, von_luschan_scale, i, tolerance)
        save_rgb_array_as_image(image_rgb_array_new, "./testimg/nambia-2-" + str(i) + "_tolerance_" + str(tolerance) + ".png")

def get_array_of_color_distance(color_array, color, save=False):
    print("get_array_of_color_distance")
    # Get the array of color distances between color_array and color
    color_distance_array = np.zeros_like(color_array, dtype=np.uint8)
    cells = color_distance_array.shape[0] * color_distance_array.shape[1] * color_distance_array.shape[2]
    for i in range(color_array.shape[0]):
        for j in range(color_array.shape[1]):
            for k in range(color_array.shape[2]):
                color_distance_array[i, j, k] = color[k] - color_array[i, j, k]

    if save:
        #get the absolute value of the color_distance_array and scale it to 255 then save it as an image
        color_distance_array = np.absolute(color_distance_array)
        color_distance_array = scale_rgb_array_to_max_distance(color_distance_array, 255)
        save_rgb_array_as_image(color_distance_array, "color_distance_array.png")


    print("average color distance: ", np.average(color_distance_array))
    print("shape of color_distance_array: ", color_distance_array.shape)
    print("color_distance_array_sample: ", color_distance_array[10,10,1])
    print("color_distance_array_sample: ", color_distance_array[10,40,1])
    return color_distance_array


def remove_background_from_image(image_rgb_array, von_luschan_scale_flat, tolerance=20, replacement_color=[0, 255, 0]):
    print("remove_background_from_image")

    # Use the color of the corner pixel as the background color
    background_color = image_rgb_array[0, 0]
    print("background_color: ", background_color)
    image_rgb_array_new = np.copy(image_rgb_array)

    # Create a mask for the background pixels
    background_mask = np.all(image_rgb_array == background_color, axis=-1)

    # Dilate the background mask to check surrounding pixels with distance tolerance
    dilated_background_mask = binary_dilation(background_mask, structure=np.ones((3, 3)), iterations=tolerance)

    # Create a mask for Von Luschan scale pixels
    von_luschan_scale_mask = np.isin(image_rgb_array, von_luschan_scale_flat).all(axis=-1)

    # Combine masks to filter out unwanted pixels
    valid_pixels_mask = ~(background_mask | von_luschan_scale_mask | dilated_background_mask)

    # Apply masks to get valid pixels and replace them with the replacement color
    image_rgb_array_new[valid_pixels_mask] = replacement_color

    return image_rgb_array_new


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
    cells = color_array.shape[0] * color_array.shape[1]
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

def save_rgb_array_as_image(rgb_array, output_path):
    print("save_rgb_array_as_image")
    image = Image.fromarray(rgb_array)
    image.save(output_path)

def save_array_as_pickle(array, output_path):
    print("save_array_as_pickle")
    import pickle

    # Save the array as a pickle
    with open(output_path, "wb") as f:
        pickle.dump(array, f)

def load_array_from_pickle(input_path):
    print("load_array_from_pickle")
    import pickle

    # Load the array from a pickle
    with open(input_path, "rb") as f:
        array = pickle.load(f)

    return array

def check_if_file_exists(file_path):
    print("check_if_file_exists")
    # Check if the file exists
    return os.path.exists(file_path)


def define_von_luschan_scale(load=False):
    print("define_von_luschan_scale")
    #if the file exists, load the von_luschan_scale from the csv
    if load:
        if check_if_file_exists(load):
            von_luschan_scale = load_array_from_pickle(load)
            return von_luschan_scale
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
    #crate numpy array of zeros with the shape of the number of colors between each pair of rgb values
    von_luschan_scale = {}
    for i in range(len(rgb_values) - 1):
        von_luschan_scale[i] = get_array_of_colors_between_two_rgb_values(rgb_values[i], rgb_values[i + 1])
        
    #if load is set, save the von_luschan_scale to a csv
    if load:
        save_array_as_pickle(von_luschan_scale, load)
        print("saved von_luschan_scale to pickle")

    return von_luschan_scale

def flatten_van_luschan_scale(load=None, von_luschan_scale=None):
    print("flatten_van_luschan_scale")
    
    # If von_luschan_scale is a file, load it instead of flattening it
    if isinstance(load, str):
        if check_if_file_exists(load):
            von_luschan_scale = load_array_from_pickle(load)
            return von_luschan_scale

    # Create a set to store unique arrays
    valid_arrays = set()

    # Iterate through each key in von_luschan_scale
    for key, value in von_luschan_scale.items():
        # Reshape the array to (n, 3) if needed
        value = value.reshape(-1, 3) if value.ndim == 2 else value

        # Use numpy unique function to get unique rows
        unique_rows = np.unique(value, axis=0)

        # Update valid_arrays with unique rows
        valid_arrays.update(map(tuple, unique_rows))

    # Convert the set back to a numpy array
    von_luschan_scale = np.array(list(valid_arrays))

    return von_luschan_scale


    #convert the list to a numpy array
    valid_arrays = np.array(valid_arrays)

    #if load is set, save the von_luschan_scale to a csv
    if isinstance(load, str):
        save_array_as_pickle(valid_arrays, load)
        print("saved von_luschan_scale to pickle")

    return valid_arrays               
        


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

def convert_rgb_to_grey(r, g, b):
    grey = 0.299 * r + 0.587 * g + 0.114 * b
    return [grey, grey, grey]

def get_bounding_box_with_flat_von_luscian(color_array, median_color_position, von_luschan_scale_flat, tolerance=100, exclude_color=None):
    rows, cols, _ = color_array.shape

    center_x, center_y = median_color_position

    # Initialize the bounding box
    bound_top_left_x = center_x
    bound_top_left_y = center_y
    bound_bottom_right_x = center_x
    bound_bottom_right_y = center_y

    for y in range(rows):
        for x in range(cols):
            r, g, b = color_array[y, x]

            # If exclude_color is set, skip the color if it matches the exclude_color
            if exclude_color is not None and np.array_equal([r, g, b], exclude_color):
                continue

            dist = distance.euclidean([x, y], [center_x, center_y])  # Use [x, y] for 2D points

            if dist < tolerance:
                if x < center_x and x < bound_top_left_x:
                    bound_top_left_x = x

                if y < center_y and y < bound_top_left_y:
                    bound_top_left_y = y

                if x > center_x and x > bound_bottom_right_x:
                    bound_bottom_right_x = x

                if y > center_y and y > bound_bottom_right_y:
                    bound_bottom_right_y = y

    return bound_top_left_x, bound_bottom_right_x, bound_top_left_y, bound_bottom_right_y, center_x, center_y




def draw_bounding_box_on_image(image_rgb_array, x1, x2, y1, y2, x_center=False, y_center=False):
    print("draw_bounding_box_on_image")

    # Ensure bounding box coordinates are within the image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_rgb_array.shape[0] - 1, x2)
    y2 = min(image_rgb_array.shape[1] - 1, y2)

    # Draw the bounding box on the image
    image_rgb_array_bounding_box = np.copy(image_rgb_array)
    image_rgb_array_bounding_box[x1:x2 + 1, y1:y2 + 1] = image_rgb_array[x1:x2 + 1, y1:y2 + 1]

    # Draw the white border
    border_thickness = 1
    image_rgb_array_bounding_box[x1:x1 + border_thickness, y1:y2 + 1] = [255, 255, 255]  # Top border
    image_rgb_array_bounding_box[x2:x2 + border_thickness + 1, y1:y2 + 1] = [255, 255, 255]  # Bottom border
    image_rgb_array_bounding_box[x1:x2 + 1, y1:y1 + border_thickness] = [255, 255, 255]  # Left border
    image_rgb_array_bounding_box[x1:x2 + 1, y2:y2 + border_thickness + 1] = [255, 255, 255]  # Right border

    # Label corners
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3  # Adjusted font size
    font_thickness = 1

    cv2.putText(image_rgb_array_bounding_box, f"({x1}, {y1})", (y1, x1), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(image_rgb_array_bounding_box, f"({x1}, {y2})", (y2, x1), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(image_rgb_array_bounding_box, f"({x2}, {y1})", (y1, x2), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(image_rgb_array_bounding_box, f"({x2}, {y2})", (y2, x2), font, font_scale, (255, 255, 255), font_thickness)

    # Draw the center of the bounding box on the image
    if x_center and y_center:
        cv2.circle(image_rgb_array_bounding_box, (x_center, y_center), radius=10, color=(255, 255, 255), thickness=1)

    return image_rgb_array_bounding_box

def draw_bounding_boxes_on_image(image_rgb_array, bounding_boxes):
    image_rgb_array_bounding_boxes = np.copy(image_rgb_array)

    for x1, x2, y1, y2, x_center, y_center in bounding_boxes:
        image_rgb_array_bounding_boxes = draw_bounding_box_on_image(image_rgb_array_bounding_boxes, x1, x2, y1, y2, x_center, y_center)

    return image_rgb_array_bounding_boxes

def add_rgb_to_all_pixels(image_array, rgb_value):
    #create a copy of the image array
    image_array_new = np.copy(image_array)
    #add the rgb value to all pixels
    image_array_new[:, :, 0] += rgb_value[0]
    image_array_new[:, :, 1] += rgb_value[1]
    image_array_new[:, :, 2] += rgb_value[2]
    return image_array_new

def mask_array_inside_bbox(array, array2, bbox):
    print("mask_array_inside_bbox")
    #replace the values in array inside the bounding box with the values in array2
    array_new = np.copy(array)
    array_new[bbox[0]:bbox[1], bbox[2]:bbox[3]] = array2[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return array_new

def replace_colors_with_closest_von_luschan_color(image_rgb_array, von_luschan_scale_flat, tolerance=100, chunk_size=1000):
    print("replace_colors_with_closest_von_luschan_color")

    height, width, channels = image_rgb_array.shape

    # Reshape the image to a 2D array of pixels
    pixels = image_rgb_array.reshape((-1, channels))

    # Calculate the number of chunks
    num_chunks = int(np.ceil(len(pixels) / chunk_size))

    # Process the image in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pixels))

        # Extract the chunk of pixels
        chunk_pixels = pixels[start_idx:end_idx]

        # Calculate the Euclidean distances for the chunk
        distances = np.linalg.norm(von_luschan_scale_flat - chunk_pixels[:, np.newaxis], axis=-1)

        # Find the index of the color with the minimum distance for the chunk
        closest_indices = np.argmin(distances, axis=1)

        # Use the indices to get the closest colors from von_luschan_scale_flat for the chunk
        closest_colors = von_luschan_scale_flat[closest_indices]

        # Replace the corresponding pixels in the chunk
        pixels[start_idx:end_idx] = closest_colors

    # Reshape the resulting array back to the original image shape
    image_rgb_array_replaced = pixels.reshape((height, width, channels))

    return image_rgb_array_replaced

def replace_colors_with_closest_von_luschan_color_from_index(image_rgb_array, von_luschan_scale, color_index, tolerance=100, chunk_size=1000):
    print("replace_colors_with_closest_von_luschan_color_from_index")

    height, width, channels = image_rgb_array.shape

    # Reshape the image to a 2D array of pixels
    pixels = image_rgb_array.reshape((-1, channels))

    # Get the color array for the specified index
    color_array = von_luschan_scale[color_index]

    # Calculate the number of chunks
    num_chunks = int(np.ceil(len(pixels) / chunk_size))

    # Process the image in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(pixels))

        # Extract the chunk of pixels
        chunk_pixels = pixels[start_idx:end_idx]

        # Calculate the Euclidean distances for the chunk
        distances = np.linalg.norm(color_array - chunk_pixels[:, np.newaxis], axis=-1)

        # Find the index of the color with the minimum distance for the chunk
        closest_indices = np.argmin(distances, axis=1)

        # Use the indices to get the closest colors from the color_array for the chunk
        closest_colors = color_array[closest_indices]

        # Replace the corresponding pixels in the chunk
        pixels[start_idx:end_idx] = closest_colors

    # Reshape the resulting array back to the original image shape
    image_rgb_array_replaced = pixels.reshape((height, width, channels))

    return image_rgb_array_replaced


def replace_skin_tone_with_target_von_luscian_scale(original_image_array, von_luschan_scale, target_von_luschan_scale_index, tolerance=100, chunk_size=1000):
    print("replace_skin_tone_with_target_von_luscian_scale")
    #get the resolution of the image
    height, width, channels = original_image_array.shape
    #get all rgb values in the image
    pixels = original_image_array.reshape((-1, channels))
    #get a array of unique rgb values in the image
    unique_pixels = np.unique(pixels, axis=0)
    print("unique_pixels: ", unique_pixels.shape)
    #get the color array for the specified index
    target_color_array = von_luschan_scale[target_von_luschan_scale_index]
    #get the unique pixels that are in the target color array
    unique_target_pixels = np.unique(target_color_array, axis=0)
    print("unique_target_pixels: ", unique_target_pixels.shape)
    #we need to scale the unique_target_pixels to the same size as unique_pixels
    #get the number of unique pixels in the image
    num_unique_pixels = unique_pixels.shape[0]
    #get the number of unique pixels in the target color array
    num_unique_target_pixels = unique_target_pixels.shape[0]
    #get the scale factor
    scale_factor = num_unique_pixels / num_unique_target_pixels
    #scale the unique_target_pixels
    scaled_unique_target_pixels = scale_rgb_array_count_to_another(unique_pixels, unique_target_pixels)
    print("scaled_unique_target_pixels: ", scaled_unique_target_pixels.shape)
    #replace the unique pixels in the image with the scaled unique_target_pixels
    image_rgb_array_replaced = replace_colors_with_closest_von_luschan_color(original_image_array, scaled_unique_target_pixels, tolerance, chunk_size)
    return image_rgb_array_replaced





#set paths
von_luschan_scale_path = "./von_luschan_scale.pickle"
von_luschan_scale_flat_path = "./von_luschan_scale_flat.pickle"

#open the image from args
image_path = sys.argv[1]
original_image_name = os.path.basename(image_path).split(".")[0]
image_rgb_array = get_image_rgb_array(image_path)

#define the von luschan scale
von_luschan_scale = define_von_luschan_scale(von_luschan_scale_path)

#flatten the von luschan scale
von_luschan_scale_flat = flatten_van_luschan_scale(von_luschan_scale_flat_path, von_luschan_scale)

#replace colors with closest von luschan color
image_rgb_array_replaced = replace_colors_with_closest_von_luschan_color(image_rgb_array, von_luschan_scale_flat, tolerance=250, chunk_size=1000)

#replace skin tone with target von luscian scale
replaced_image = replace_skin_tone_with_target_von_luscian_scale(image_rgb_array, von_luschan_scale, 10)

#save the replaced image
save_rgb_array_as_image(replaced_image, original_image_name + "_replaced.png")