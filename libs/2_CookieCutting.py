# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:51:21 2023

@author: riyad
"""

import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import os

def save_image_with_centroid(raster, centroid, window_size_w, window_size_h, output_path_template, point_id):
    # Convert centroid coordinates to pixel coordinates
    pixel_coords = raster.index(centroid.x, centroid.y)
    
    # Extract a window around the centroid
    window = raster.read(window=((pixel_coords[0] - window_size_w // 2, pixel_coords[0] + window_size_w // 2),
                                 (pixel_coords[1] - window_size_h // 2, pixel_coords[1] + window_size_h // 2)))

    # Check if the window size is valid
    if window.shape[1] == 0 or window.shape[2] == 0:
        print(f"Skipping centroid due to invalid window size.")
        return

    # Create the plot
    plt.imshow(window.transpose(1, 2, 0))
    plt.scatter([window_size_w // 2], [window_size_h // 2], s=100, facecolors='none', edgecolors='yellow', linewidth=2, marker='o')
    plt.axis('off')  # Turn off axes

    # Save the plot as an image file
    os.makedirs(os.path.dirname(output_path_template), exist_ok=True)
    output_path = f"{output_path_template}point_{point_id}.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save without extra whitespace

    # Close the plot to release resources
    plt.close()
    return output_path

def save_images(points_path, rasters_path, window_size_w, window_size_h, output_path_template):
    import os

    # Read the vector data (tree points)
    points = gpd.read_file(points_path)

    # List all files in the raster directory that have a '.tif' extension
    tif_files = [f for f in os.listdir(rasters_path) if f.endswith('.tif')]

    # Iterate through each TIFF file in the raster directory
    for tif_file in tif_files:
        grid_id = tif_file.split('_')[1].split('.')[0]
        tif_path = os.path.join(rasters_path, tif_file)
        
        # Open the raster file using rasterio
        with rio.open(tif_path) as raster:
            points_in_grid = points[points['grid_id'] == int(grid_id)]
            
            # Iterate through each tree point in the filtered subset
            for index, row in points_in_grid.iterrows():
                # Get the centroid of the tree point
                centroid = row.geometry
                
                # Save the image with highlighted centroid
                path = save_image_with_centroid(raster, centroid, window_size_w, window_size_h, output_path_template, row['id'])
                print(f'Point: {row["id"]}.   Path: {path}')

# Example usage
training_raster_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TrainingData/Train_GoogleImages'                
training_vector_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TrainingData/Train_400_point_per_grid_with_400_m_global_distance_after_30meters_buffer_3857.gpkg'
training_rgb_save_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TrainingData/Train_GoogleImages/PixelAroundPoints/'


testing_raster_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TestingData/Test_GoogleImages'
testing_vector_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TestingData/Test_400_point_per_grid_with_400_m_global_distance_after_30meters_buffer_3857.gpkg'
testing_rgb_save_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TestingData/Test_GoogleImages/PixelAroundPoints/'


save_images(testing_vector_path, testing_raster_path, 256, 256, testing_rgb_save_path)

save_images(training_vector_path, training_raster_path,256, 256, training_rgb_save_path)
