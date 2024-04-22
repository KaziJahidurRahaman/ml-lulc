# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:25:14 2023

@author: riyad
"""

class FeatureGeneration:
    def __init__(self, *args, **kwargs):
        pass
    
    # ndbi
    def functn_Ndbi(SWIR, NIR):
        'Function to calculate ndbi for the provided image. Takes the multiband image as input, and returns a single band image \
        as output. NDBI formula is (B11 -B8)/(B11+B8). collected from https://d-nb.info/1195147821/34'

        ndbi = (SWIR - NIR)/(SWIR+NIR) 
        return ndbi

    #NDWI
    def functn_Ndwi(GREEN, NIR):
        'Calculates the NDWI for. Take the multiband image as input, returns single band NDWI image.\
        NDWI formula NDWI= (Band 3 – Band 8)/(Band 3 + Band 8) is collected from doi:10.1080/01431169608948714'
        
        ndwi = ((GREEN - NIR)/(GREEN+NIR)) # in sentinel 2 b3 is Green and b8 is NIR
        return ndwi

    # NDVI
    def functn_Ndvi(NIR, RED):
        'Calculates the NDVI. Takes the multiband image as input and returns single band NDVI image\
        Formula is NDVI = (B8-B4)/(B8+B4) collected from https://www.geo.fu-berlin.de/en/v/geo-it/gee/2-monitoring-ndvi-nbr/2-1-basic-information/index.html'

        ndvi = ((NIR - RED)/(NIR+RED)) # In Sentinel 2 B8 is NIR and B4 is Red
        return ndvi

    # BSI
    def functn_Bsi(BLUE, RED, NIR, SWIR2):
        'Calculates the Baresoil index. Take the multiband image as input and returns singleband BSI image\
        The formula BSI = ((SWIR2 + RED)−(NIR + BLUE)) / ((SWIR2 + RED)+(NIR + BLUE)). Collected from Land 2021, 10(3), 231; https://doi.org/10.3390/land10030231'
        
        bsi = ( ( SWIR2 + RED ) - ( NIR + BLUE ) ) / ( ( SWIR2 + RED ) + ( NIR + BLUE))
        return bsi 
        
    # NDSI    (B3-B11)/(B3+B11)   (Green Band - SWIR Band) / (Green Band + SWIR Band)
    def functn_Ndsi(img):
        import ee
        'Calculates the NDSI. Take the multiband image as input, returns single band NDSI image.'
        ndsi = img.normalizedDifference(['B3', 'B11']).rename('NDSI')
        return ndsi
    
    # EVI     2.5 * ((B8 – B4) / (B8 + 6 * B4 – 7.5 * B2 + 1))
    def functn_Evi(image):
        import ee
        coef1 = ee.Number(2.5)
        coef_red = ee.Number(6)
        coef_blue = ee.Number(7.5)
        const = ee.Number(1)
    
        # compute EVI
        evi = image.expression('coef1*((nir-red)/(nir + (coef_red*red)-(coef_blue*blue) + const))',{
            'nir': image.select('B8'),
            'red': image.select('B4'),
            'blue': image.select('B2'),
            'coef1':coef1,
            'coef_red':coef_red,
            'coef_blue':coef_blue,
            'const':const}).rename('EVI')
        return evi
    
    
    # NBR     (B8-B12)/(B8+B12)  (NIR - SWIR) / (NIR + SWIR) DOI: 10.1080/10106049109354290
    def functn_Nbr(img):
        import ee
        'Calculates the NBR. Take the multiband image as input, returns single band NBR image.\
        NBR formula (B8-B12)/(B8+B12)  (NIR - SWIR) / (NIR + SWIR) is collected from doi:10.1080/01431169608948714'
        nbr = img.normalizedDifference(['B8', 'B12']).rename('NBR') # in sentinel 2 b8 is NIR and b12 is SWIR
        return nbr
    
    