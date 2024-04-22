# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:30:07 2023

@author: riyad
"""

class Preprocessing:
    
    def __init__(self, *args, **kwargs):
        pass
    
    # Functions
    def getCloudMasks(image):
        import ee
        scl = image.select('SCL')
    
        #Selecting Cloudy Mask
        image_cloud_shadow = image.select('SCL').eq([3])
        image_cloud_low = image.select('SCL').eq([7])
        image_cloud_med = image.select('SCL').eq([8])
        image_cloud_high = image.select('SCL').eq([9])
        image_cloud_cirrus = image.select('SCL').eq([10])
    
    
        cloud_mask = image_cloud_shadow.add(image_cloud_low).add(image_cloud_med).add(image_cloud_high).add(image_cloud_cirrus)
    
    #     #Inverting the Mask
        invert_cloud_mask = cloud_mask.eq(0).selfMask().rename('NO_CLOUD_MASK') #invert mask will have the pixels with only no cloud
        cloud_mask_only = cloud_mask.eq(1).selfMask().rename('ONLY_CLOUDS_MASK') #this will have only the pixels without cloud
    
        masks = invert_cloud_mask.addBands(cloud_mask_only)
        return masks
    
    #     image_only_clouds =  image.updateMask(cloud_mask_only).divide(10000)
    #     stats = image_only_clouds.reduceRegion(reducer= ee.Reducer.sum(), geometry= aoi_ee, scale= 10)
    #     print(stats.getInfo())
    
    # #     img_masked = img_cloudy.updateMask(invert_mask)
    # #     img_unmasked = img_masked.unmask(-1)
    # #     img_cloudy_cloudless = img_unmasked
    
    #     image_cloudless = image.updateMask(invert_cloud_mask).divide(10000)
    # #     fill_image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/'+fill_image_id)
    # #     img_cloudy_cloudless = img_cloudy_cloudless.unmask(fill_image).divide(10000)
    
    #     return image_cloudless
    
    
    def cloudFIll(image, filler):
        import ee
        filler = ee.Image(filler) #.divide (10000)
        image_filled = image.unmask(filler)
    
        return image_filled
    
    
    def functn_scale_bands(image, bandstoscale, scalefactor):
        import ee
        scaledbands = image.select(bandstoscale).multiply(scale_factor)
    
        return image.addBands(scaledbands, overwrite=True)
    
    def functn_ResemapleSentinel2(img, sampling_algorithm,  scale):
        'Function to resample the sentinel bands from there native scale to 10 meter scale. Takes the image as iput, return the resampled image'
        
        import ee
        crs =  img.select('B1').projection().crs()
        img = img.resample(sampling_algorithm).reproject(crs=crs, scale=scale)
        return img
    
    
    def functn_Clip(img, aoi_ee):
        'Cliping the bands of the image to the area of interest. Takes the image and aoi feature as input and returns the clipped image.'
        
        import ee
        clipped_img = img.clip(aoi_ee)
        return clipped_img
    

    def imagePrediction(model, bandsData, classifier):
        '''Function to classify the complete image
        inputs: model- model to use to classify the image
                bandsData- the image that is to classify
                fclass- preffered class in visualisation. 
        output: predicted_Image
        '''
        import ee
        import numpy as np
        from tqdm import tqdm
        
        predicted_LULC=[]
        column_length=bandsData[0].shape[1]
        
        for i in tqdm(range(bandsData[0].shape[0])):
            spectrum_data_at_row_i=[]
            for m in range(len(bandsData)):
                spectrum_data_at_row_i.append(bandsData[m][i])
            spectrum_data_at_row_i_T=np.transpose(np.array(spectrum_data_at_row_i))
            
            if classifier=='RF' or classifier=='ANN':
                predicted_class=model.predict(spectrum_data_at_row_i_T)
                predicted_LULC.append(np.argmax(predicted_class, axis = 1))
            else:
                predicted_class=np.transpose(model.predict(spectrum_data_at_row_i_T))
                predicted_LULC.append(predicted_class)
    
        predicted_LULC=np.array(predicted_LULC).astype(np.uint8)
     
        return predicted_LULC
    
    def preprocessShapeFiles(gpd_file,BandsCrs ):
        import geopandas as gpd
        
        
        if(gpd_file.crs is None):
            print(f'Training shape file contains no CRS. Setting CRS {BandsCrs}')
            gpd_file = gpd_file.set_crs(BandsCrs, allow_override=True)
        elif gpd_file.crs != BandsCrs:
            print(f'Training and Band CRS Doesnot Match. Training Data CRS Reprojecting to {BandsCrs}..')
            gpd_file = gpd_file.to_crs(BandsCrs)
   
        # gpd_file.columns = ['pointid_in_layer','layer_name','lclass','index', 'geometry'] # change the column name id to point_id
        # gpd_file['index'] = gpd_file.index #create a new column index
        # gpd_file['id'] = gpd_file.index #create a new column index

        # gpd_file.loc[gpd_file['lclass'] == 8, 'lclass'] = 3
        preprocessed_gpd = gpd_file
        
        return preprocessed_gpd
        
