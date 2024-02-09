# Density map

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from wand.api import library
from wand.image import Image
from wand.color import Color


def process_data(tsv_file_path, image_size):
    # Step 1: Read the TSV file
    data = np.loadtxt(tsv_file_path, delimiter='\t', skiprows = 1, usecols = (5,6))
    # Remove rows with negative values in x or y
    data = data[(data[:, 0] >= 0) & (data[:, 1] >= 0)]

    # Remove rows with coordinates outside image boundaries
    data = data[(data[:, 0] >= 0) & (data[:, 0] < image_size[1]) & (data[:, 1] >= 0) & (data[:, 1] < image_size[0])]

    return data

def create_density_map(tsv_file_path, output_path_density_map, path_referenceImage, output_path_expansion_map):

    #Read the image file
    image = cv2.imread(path_referenceImage)
    image_size = image.shape 

    data = process_data(tsv_file_path, image_size)

    # Step 2: Create a Density Map
    density_map_orig  = np.zeros(image_size[:-1])
    density_map = np.zeros(image_size)

    # Define the radius of the circle
    radius = 60

    for point in data:
        (x, y) = point

        # Convert (x, y) to integer coordinates
        x_int, y_int = int(x), int(y)

        # Update density map for the tracked position itself
        density_map_orig[y_int, x_int] += 1  
        density_map[y_int, x_int] += 1

        # Update density map for the neighboring pixels within the radius
        for i in range(max(0, y_int - radius), min(image_size[0], y_int + radius + 1)):
            for j in range(max(0, x_int - radius), min(image_size[1], x_int + radius + 1)):
                distance = np.sqrt((x_int - j)**2 + (y_int - i)**2)
                if distance <= radius:
                    density_map[i, j] += 1


    # # Step 3: Overlay the Density Map on the Image
    # # Normalize the density map for overlay
    # density_map_normalized = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

    # # Convert density map to 3 channels (for RGB)
    # density_map_colored = cv2.applyColorMap(np.uint8(density_map_normalized), cv2.COLORMAP_JET)

    # # Overlay the density map on top of the image with transparency
    # overlay = cv2.addWeighted(image, 0.5, density_map_colored, 0.5, 0)

    # # Save the overlaid image
    # cv2.imwrite(output_path_density_map , overlay)

    # Creating the expansion map
    inflation_coords = find_inflation_points(density_map_orig)
    (height, width) = density_map_orig.shape
    create_inflation_map(image, path_referenceImage, inflation_coords, output_path_expansion_map, height, width)


def project_array_to_pixel(coords ,step, square_size, height, width):
    y,x = coords
    if len(x)>1:
        return np.transpose(np.array([(min(height, y_val*step)+square_size/2, min(width, x_val*step)+square_size/2) for y_val,x_val in zip(y,x)]))
    else:
        return np.array((min(height, y[0]*step)+square_size/2, min(width, x[0]*step)+square_size/2))



def find_inflation_points(density_map_orig):

    (height, width) = density_map_orig.shape
    print("Size of the ref image: ", density_map_orig.shape)
    divison_factor = 20
    square_size = int(min(width/divison_factor, height/divison_factor))
    width_resid = square_size - width%square_size
    height_resid = square_size - height%square_size

    # Padd the array
    padd_rows = np.zeros((height_resid, width)) 
    padd_columns = np.zeros((height+height_resid, width_resid))
    padded_map = np.hstack((np.vstack((density_map_orig, padd_rows)), padd_columns))

    # Go over the array with window of square size
    step = int(square_size/2)
    smaller_height = int(padded_map.shape[0]/step)
    smaller_width = int(padded_map.shape[1]/step)
    smaller_map = np.zeros((smaller_height, smaller_width))
    smaller_map_modif = np.copy(smaller_map)
    print("Smaller map size: ", smaller_map.shape)
    # How far the values should be included
    smaller_window = round(divison_factor/7) #* int(square_size/step) 
    print(f"How far you look: {smaller_window} or square of {(smaller_window*2+1)*step} pixel size ")

    
    #Iterate vertically and horizontally over an array
    for y in range(0, height, step):
        for x in range(0, width, step):
            window_sum = np.sum(padded_map[y:y+square_size, x:x+square_size])
            smaller_map[round(y/step), round(x/step)] = window_sum

    # Process smaller map
    # Get the unique values from the map exluding 0
    uniq_vals = np.array(sorted(np.unique(smaller_map)))
    uniq_vals = uniq_vals[uniq_vals != 0]

    # Iterate through and find their coordinates in the smaller map 
    for val in uniq_vals:
        coords_indices = np.where(smaller_map==val)
        coords_indices_list = list(zip(coords_indices[0], coords_indices[1]))

        #
        for (y,x) in coords_indices_list:
            
            cutted_map = np.copy(smaller_map[max(0, y-smaller_window):min(smaller_height,y+smaller_window), max(x-smaller_window,0):min(smaller_width, x+smaller_window)])
            
            # Create copies for finding highest neighbour around current position
            smaller_map_copy = np.copy(smaller_map)
            smaller_map_copy[y,x]=0
            cutted_map_copy = smaller_map_copy[max(0, y-smaller_window):min(smaller_height,y+smaller_window), max(x-smaller_window,0):min(smaller_width, x+smaller_window)]
            highest_neighbor = np.max(cutted_map_copy)

            # If it is smaller than the highest neighbour, put zero in new map at its position
            if smaller_map[y,x] < highest_neighbor:
                smaller_map_modif[y,x] = 0
            # If higher, sum neighbourhood and put it in new map at its position
            elif smaller_map[y,x] > highest_neighbor:
                smaller_map_modif[y,x] = np.sum(cutted_map)
            # If it is equal to the highest neighbour, put zero in new map at its position, sum neighbourhood and put it at mean position between highest neigbours
            else:
                temp_map = np.zeros(padded_map.shape)
                temp_map[max(0, y-smaller_window):min(smaller_height,y+smaller_window), max(x-smaller_window,0):min(smaller_width, x+smaller_window)] = np.copy(cutted_map)
                row_inds, col_inds= np.where(temp_map==val)
                cutted_y = round(np.mean(row_inds))
                cutted_x = round(np.mean(col_inds))
                smaller_map_modif[y,x] = 0
                smaller_map_modif[cutted_y,cutted_x] = np.sum(cutted_map)

    # Filter out values less than threshold (% of max value)
    highest_val = np.max(smaller_map_modif)
    smaller_map_modif[smaller_map_modif < 0.5 * highest_val] = 0

    # Get the coordinates and magnitude for creating expanding effect on the image
    inflation_points = []
    for val in sorted(np.unique(smaller_map_modif), reverse = True):
        if val!=0:
            inflation_points.append([val, project_array_to_pixel(np.where(smaller_map_modif==val), step, square_size,height, width)])
    print(inflation_points)

    return inflation_points



def create_inflation_map(image, path_referenceImage, inflation_coords, output_path_expansion_map, height, width):

    with Image(filename=path_referenceImage) as img:
        # Grab image size
        width, height = img.size #cols - x, rows - y

        # Calculate the increase in size (10% on each side)
        increase_width = int(width* 0.2)
        increase_height = int(height * 0.2)

        # Create a new image with increased size and white background
        new_width = width + 2 * increase_width
        new_height = height + 2 * increase_height
        new_image = Image(width=new_width, height=new_height, background=Color("white"))

        ratio_size =  width/new_width * 0.85
        print(ratio_size)

        # Composite the original image onto the new image
        composite_position = (increase_width, increase_height)
        new_image.composite(img, composite_position[0], composite_position[1])
        img = new_image

        highest_magn = inflation_coords[-1][0]

        for i, (magnitude, coords) in enumerate(inflation_coords):
            y,x = coords
            
            # If several points of the same magnitude is present
            if coords.shape!=(2,):
                for y_coord, x_coord in zip(y, x):
                        #print(f"{y_coord/height:.3f},{x_coord/width:.3f}")

                        # Define the target location 
                        ty, tx = int(y_coord)+increase_height, int(y_coord)+increase_width
                        # Find middle of the image.
                        mx, my = new_width // 2, new_height // 2
                        img.options['filter'] = 'point'
                        # Increase viewpoint to allow pixels to move out-of-bounds.
                        viewport = '{0}x{1}+{2}+{3}'.format(tx*2+new_width, ty*2+new_height, tx, ty)
                        img.options['distort:viewpoint'] = viewport
                        img.virtual_pixel = 'tile'
                        # Distort target coords to middle of image.
                        img.distort(method='scale_rotate_translate',
                                    arguments=[tx, ty, 1, 0, mx, my])
                        # Implode
                        img.implode(amount=-magnitude/highest_magn * ratio_size) #, method = 'average')
                        # Distort middle back into place.
                        img.distort(method='scale_rotate_translate',
                                    arguments=[mx, my, 1, 0, tx, ty])
                        # Restore original image size.
                        img.extent(width=new_width, height=new_height)
                        # done
                        #img.save(filename=output_path_expansion_map)

            # If there is only single value of correspnding magnitude
            else:
                    # Define the target location 
                    ty, tx = int(y)+increase_height, int(x)+increase_width
                    # Find middle of the image.
                    mx, my = new_width // 2, new_height // 2
                    img.options['filter'] = 'point'
                    # Increase viewpoint to allow pixels to move out-of-bounds.
                    viewport = '{0}x{1}+{2}+{3}'.format(tx*2+new_width, ty*2+new_height, tx, ty)
                    img.options['distort:viewpoint'] = viewport
                    img.virtual_pixel = 'tile'
                    # Distort target coords to middle of image.
                    img.distort(method='scale_rotate_translate',
                                arguments=[tx, ty, 1, 0, mx, my])
                    # Implode
                    img.implode(amount=-magnitude/highest_magn * ratio_size )#,method = 'average')
                    # Distort middle back into place.
                    img.distort(method='scale_rotate_translate',
                                arguments=[mx, my, 1, 0, tx, ty])
                    # Restore original image size.
                    img.extent(width=new_width, height=new_height)
                    # Save the image 
                    #img.save(filename=output_path_expansion_map)

        img.save(filename=output_path_expansion_map)

# def create_inflation_map(image, path_referenceImage, inflation_coords, output_path_expansion_map, height, width):

#         highest_magn = inflation_coords[-1][0]
#         for i, (magnitude, coords) in enumerate(inflation_coords):
#             if i == 0:
#                 file_path = path_referenceImage
#             else:
#                 file_path = output_path_expansion_map

#             y,x = coords
#             if coords.shape!=(2,):
#                 for y_coord, x_coord in zip(y, x):
#                     with Image(filename=file_path) as img:
#                         #print(f"{y_coord/height:.3f},{x_coord/width:.3f}")

#                         # with Image(filename=path_referenceImage) as img:
#                         # Grab image size
#                         cols, rows = img.size
#                         # Define our target location ... say 1/3rd, by 1/5th
#                         ty, tx = int(y_coord), int(y_coord)
#                         # Find middle of the image.
#                         mx, my = cols // 2, rows // 2
#                         # Roll target coord into middle
#                         ok = library.MagickRollImage(img.wand, mx-tx, my-ty)
#                         if not ok:
#                             img.raise_exception()
#                         # Implode
#                         img.implode(-magnitude/highest_magn)
#                         # Roll middle back into place.
#                         ok = library.MagickRollImage(img.wand, mx+tx, my+ty)
#                         if not ok:
#                             img.raise_exception()
#                         # done
#                         img.save(filename=output_path_expansion_map)

#             else:
#                 with Image(filename=file_path) as img:
#                     #print(f"{y/height:.3f}, {x/width:.3f}")

#                     # with Image(filename=path_referenceImage) as img:
#                     # Grab image size
#                     cols, rows = img.size
#                     # Define our target location ... say 1/3rd, by 1/5th
#                     ty, tx = int(y), int(x)
#                     # Find middle of the image.
#                     mx, my = cols // 2, rows // 2
#                     # Roll target coord into middle
#                     ok = library.MagickRollImage(img.wand, mx-tx, my-ty)
#                     if not ok:
#                         img.raise_exception()
#                     # Implode
#                     img.implode(-magnitude/highest_magn)
#                     # Roll middle back into place.
#                     ok = library.MagickRollImage(img.wand, mx+tx, my+ty)
#                     if not ok:
#                         img.raise_exception()
#                     # done
#                     img.save(filename=output_path_expansion_map)




# def create_line_map(data):

#     # Convert the list of points to a NumPy array
#     data_int = np.array(data, np.int32)

#     # Reshape the array to a 1xN array (required by cv2.polylines)
#     pts_array = data_int.reshape((-1, 1, 2))

#     image_with_circles = overlay.copy()

#     # Connect the points with lines on the image
#     color = (0, 255, 0)  # Green color
#     thickness = 3
#     image_with_lines = cv2.polylines(image_with_circles, [pts_array], isClosed=False, color=color, thickness=thickness)

#     # # Add arrows to show the direction of the lines
#     # thickness_arrow = 3
#     # for i in range(len(data_int) - 1):
#     #     cv2.arrowedLine(image_with_lines, tuple(data_int[i]), tuple(data_int[i + 1]), color, thickness=thickness_arrow , tipLength=0.025)

#     # # Display the image with connected points, circles, and arrows
#     # cv2.imshow("Connected Points with Circles and Arrows", image_with_lines)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


#     # # Draw growing circles at each point
#     # r_max = 40
#     # r_step = r_max/len(data_int)
#     # r = 1
#     # for i, point in enumerate(data_int):
#     #     r = (r + r_step)
#     #     cv2.circle(image_with_lines, tuple(point), int(r), color, thickness=-1)

#     # Highlight the first and last points with different color circles
#     cv2.circle(image_with_circles, tuple(data_int[0]), 15, (0, 0, 255), -1)  
#     cv2.circle(image_with_circles, tuple(data_int[-1]), 30, (0, 0, 255), -1)  

#     # Display the image with connected points, circles, and arrows
#     cv2.imshow("Connected Points with Growing Circles", image_with_lines)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputRoot', help='Path to result_outputs directory')
    args = parser.parse_args()

    #Iterate through folders
    folder_path = os.path.join(args.outputRoot, 'Processed_mapped')

    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)

        if os.path.isdir(subfolder_path):

            #File paths
            tsv_file_path  = os.path.join(subfolder_path, 'gazeData_mapped.tsv')
            # List all files in the folder
            all_files = os.listdir(subfolder_path)
            referenceImage_file = [os.path.join(subfolder_path, file) for file in all_files if (file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'))]
            path_referenceImage = referenceImage_file[0]

            path_output = os.path.join(args.outputRoot, 'Final_results', folder_name)
            output_path_density_map = os.path.join(path_output,  f'density_map_{folder_name}.jpeg')
            output_path_expansion_map = os.path.join(path_output,  f'expansion_map_{folder_name}.jpeg')

            print(f"Creating a density map for folder {folder_name}")
            create_density_map(tsv_file_path, output_path_density_map, path_referenceImage, output_path_expansion_map)
            #create_line_map(tsv_file_path, output_path_density_map)




