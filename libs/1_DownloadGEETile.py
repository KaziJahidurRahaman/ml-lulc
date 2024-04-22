# # Commented out IPython magic to ensure Python compatibility.
# # %pip -q install segment-geospatial

# import leafmap
# from samgeo import tms_to_geotiff
# from samgeo import SamGeo
# # from google.colab import drive
# # # drive.mount('/content/drive')

# """## Create an interactive map"""

# m = leafmap.Map(center=[11.560975, 104.892177], zoom=19, height="800px")
# m.add_basemap("Esri.WorldImagery")

# import geopandas
# from shapely.geometry import box
# import os

# grid_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TrainingData/TrainGrids4326.gpkg'
# grids = geopandas.read_file(grid_path)
# # Specify the target CRS (WGS 84)
# target_crs = 'EPSG:4326'

# """## Download a sample image

# Pan and zoom the map to select the area of interest. Use the draw tools to draw a polygon or rectangle on the map
# """

# for index, row in grids.iterrows():
#     # Access the 'id' column in each row
#     grid_id = row['id']
#     grid_left = row['left']
#     grid_right = row['right']
#     grid_top = row['top']
#     grid_bottom = row['bottom']

#     grid_geometry = row.geometry

#     # Convert the bounding box of the grid geometry to the target CRS (WGS 84)
#     bounding_box_wgs84 = grid_geometry.bounds
#     bounding_box_wgs84 = geopandas.GeoSeries(box(*bounding_box_wgs84), crs=grids.crs)
#     bounding_box_wgs84 = bounding_box_wgs84.to_crs(target_crs)

#     # Access the bounding box information in WGS 84
#     bounding_box_wgs84 = list(bounding_box_wgs84.geometry[0].bounds)

#      # Create a directory path for each grid if it doesn't exist
#     output_directory = f"allImages/Train/Image_{grid_id}"
#     if not os.path.exists(output_directory):os.makedirs(output_directory)
#     image = f"{output_directory}/Image_{grid_id}.tif"
#     tms_to_geotiff(output=image, bbox=bounding_box_wgs84, zoom=18, source="Esri.WorldImagery", overwrite=True)


# Commented out IPython magic to ensure Python compatibility.
# %pip -q install segment-geospatial

from samgeo import tms_to_geotiff
from samgeo import SamGeo



def download_sample_images(grids_path, basemap_source , zoom,  target_crs='EPSG:4326', output_base_directory='allImages/'):
    import os
    import geopandas
    from shapely.geometry import box
    
    grids = geopandas.read_file(grids_path)
    
    """
    Download sample images based on the specified grid polygons.

    Parameters:
    - grids: GeoDataFrame
        GeoDataFrame containing grid polygons.
    - target_crs: str, optional
        Target Coordinate Reference System (CRS). Default is 'EPSG:4326'.
    - output_base_directory: str, optional
        Base directory where images will be saved. Default is 'allImages/Train'.

    Returns:
    - None
    """

    # Create the output base directory if it doesn't exist
    if not os.path.exists(output_base_directory):
        os.makedirs(output_base_directory)

    for index, row in grids.iterrows():
        # Access the 'id' column in each row
        grid_id = row['id']
        grid_geometry = row.geometry

        # Convert the bounding box of the grid geometry to the target CRS (WGS 84)
        bounding_box_wgs84 = grid_geometry.bounds
        bounding_box_wgs84 = geopandas.GeoSeries(box(*bounding_box_wgs84), crs=grids.crs)
        bounding_box_wgs84 = bounding_box_wgs84.to_crs(target_crs)

        # Access the bounding box information in WGS 84
        bounding_box_wgs84 = list(bounding_box_wgs84.geometry[0].bounds)

        # Create a directory path for each grid if it doesn't exist
        output_directory = os.path.join(output_base_directory, f"Image_{grid_id}")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Specify the image file path
        image_path = os.path.join(output_directory, f"Image_{grid_id}.tif")

        # Download the image using tms_to_geotiff
        tms_to_geotiff(output=image_path, bbox=bounding_box_wgs84, zoom=zoom, source=basemap_source,  overwrite=True)


# Example usage:
# Assuming 'grids' is the GeoDataFrame containing grid polygons
# download_sample_images(grids, target_crs='EPSG:4326', output_base_directory='allImages/Train')
train_grids_path =  'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TrainingData/TrainGrids4326_2.gpkg'
# test_grids_path = 'D:/3. Projects/ClassificationTasks/Classification-No74Indices/AfterReviewFromProfessor/TestingData/TestGrids4326.gpkg'


download_sample_images(grids_path = train_grids_path, basemap_source ="Satellite" , zoom = 19,   target_crs='EPSG:4326', output_base_directory='TrainingData/Train_GoogleImages')
# download_sample_images(grids_path = test_grids_path, basemap_source = "Satellite" , zoom = 19,  target_crs='EPSG:4326', output_base_directory='TestingData/Test_GoogleImages')
