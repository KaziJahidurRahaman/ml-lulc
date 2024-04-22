# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:41:24 2023

@author: riyad
"""

class Utilities:
    def __init__(self, *args, **kwargs):
        pass
    
    
    def loadEE(GEE_SERVICE_ACC, GEE_TOKEN_PATH):
        "Function Description"
        import ee
        
        credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACC, GEE_TOKEN_PATH)
        try:
            ee.Initialize(credentials, opt_url='https://earthengine-highvolume.googleapis.com')
            return 'Authentication successful'
        except:
            return'Authentication Failed'
            
            
    def loadTrainingImage(gee_arch, mgrs_tile, aoi_path, f_year, f_month, t_year, t_month, max_cloud, dark_feature_percenatge):
        "Function Description"
        import ee
        import geemap
         
        aoi_ee = geemap.shp_to_ee(aoi_path)
        allfilteredImages = ee.ImageCollection(gee_arch).filterMetadata('MGRS_TILE', 'equals', mgrs_tile).filterBounds(aoi_ee).\
        filter(ee.Filter.calendarRange(f_year,t_year,'year')).filter(ee.Filter.calendarRange(f_month,t_month,'month')).\
        filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud)).filter(ee.Filter.lt('DARK_FEATURES_PERCENTAGE', dark_feature_percenatge))  # We are filtering the dark feature with a threshold of 1
                                                            #to reduce time we are checking the image of only on the tiles where aoi located      
        bestCloudImage = allfilteredImages.sort('CLOUDY_PIXEL_PERCENTAGE',True).first() # Sorting the image on cloudcover and taking the best image       
        bestCloudImageprojection =  bestCloudImage.select('B1').projection()
        bestCloudImagecrs = bestCloudImageprojection.crs()
        bestCloudImagetransform_image = bestCloudImageprojection.transform()
        return bestCloudImage, bestCloudImageprojection, bestCloudImagecrs, bestCloudImagetransform_image
    
    
    def loadFillerImage(gee_arch, mgrs_tile, aoi_path, f_year, f_month, t_year, t_month, trainingImageProjection):
        "Function Description"
        import ee
        import geemap
        
        aoi_ee = geemap.shp_to_ee(aoi_path)
        
        allfilteredImages = ee.ImageCollection(gee_arch).filterMetadata('MGRS_TILE', 'equals', mgrs_tile).filterBounds(aoi_ee).\
        filter(ee.Filter.calendarRange(f_year,t_year,'year')).filter(ee.Filter.calendarRange(f_month,t_month,'month')).\
        filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50)).filter(ee.Filter.lt('DARK_FEATURES_PERCENTAGE', 50))  # We are filtering the dark feature with a threshold of 1
                                                            #to reduce time we are checking the image of only on the tiles where aoi located
        count = allfilteredImages.size().getInfo()
        print(f'Image Used for filler image: {count}')
        fillerImage = allfilteredImages.median().reproject(trainingImageProjection)
        return fillerImage
    
    
    def preprocessImage(gee_image, fillerImage):
        import ee
        from .ee_preprocessing import Preprocessing as Preprocessor
       
        #Generating Cloud Masks (Low, Medium, High Cloud, CLoud Shadow, Cirus)
        cloudMasks = Preprocessor.getCloudMasks(gee_image) # applying the cloud mask
        nocloudMask = cloudMasks.select('NO_CLOUD_MASK')
        onlycloudsMask = cloudMasks.select('ONLY_CLOUDS_MASK')
        #Masking the image and Scaling
        ImageOnly_clouds =  gee_image.updateMask(onlycloudsMask)#.divide(10000)
        ImageCloudMasked = gee_image.updateMask(nocloudMask)#.divide(10000)
        # CloudFilling
        ImageCloudMaskedAndFilled = Preprocessor.cloudFIll(ImageCloudMasked,fillerImage) # filling the cloud removed pixel with
        # Resampling
        ImageCloudMaskedAndFilledResampled = Preprocessor.functn_ResemapleSentinel2(ImageCloudMaskedAndFilled, 'bilinear', 10) #reseampled to 10m
        print('Preprocessing Completed. Returning the processed image...')
        return ImageCloudMaskedAndFilledResampled
    
    
    def generateFeature(gee_image):
        from .ee_featuregeneration import FeatureGeneration as FeatureGenerator
        
        ndbi = FeatureGenerator.functn_Ndbi(gee_image)
        ndwi = FeatureGenerator.functn_Ndwi(gee_image)
        ndvi = FeatureGenerator.functn_Ndvi(gee_image)
        bsi = FeatureGenerator.functn_Bsi(gee_image)
        ndsi = FeatureGenerator.functn_Ndsi(gee_image)
        evi = FeatureGenerator.functn_Evi(gee_image)
        nbr = FeatureGenerator.functn_Nbr(gee_image)
        gee_image_with_added_feature = gee_image.addBands(ndbi).addBands(ndwi).addBands(ndvi).addBands(bsi).addBands(ndsi).addBands(evi).addBands(nbr)
        print('Feature Engineering Completed. Returning the image...')
        return gee_image_with_added_feature


    
    def trimImage(gee_image, aoi_path, bands_to_include):
        import geemap
        from .ee_preprocessing import Preprocessing as Preprocessor
       
        aoi_ee = geemap.shp_to_ee(aoi_path)
        clippedImage = Preprocessor.functn_Clip(gee_image, aoi_ee) 
        trimmedImage = clippedImage.select(bands_to_include)
        trimmedImage = trimmedImage.toFloat()
      
        projection_trimmedImage =  trimmedImage.select('B1').projection()
        crs_trimmedImage = projection_trimmedImage.crs()
        transform_trimmedImage = projection_trimmedImage.transform()
        print('Clipping and Band selection completed. Returning trimmed image')
        return trimmedImage, projection_trimmedImage, crs_trimmedImage, transform_trimmedImage

    
    
    # def downloadBands(image, selectedbands, aoi_path, export_path, scale):
    #     import concurrent.futures
    #     import geemap
    #     import os
    #     from retry import retry
        
    #     aoi_ee = geemap.shp_to_ee(aoi_path)
    #     urls = {}
        
    #     @retry(tries=10, delay=1, backoff=2)
    #     def download_band(url, band_name):
    #         import requests
    #         try:
    #             response = requests.get(url, stream=True)
    #             if response.status_code == 200:
    #                 temp_path = f'{export_path}/Bands/'
    #                 if not os.path.exists(temp_path):os.makedirs(temp_path)
    #                 temp_path = f'{temp_path}{band_name}.tif' 
    #                 with open(temp_path, 'wb') as file:
    #                     file.write(response.content)
    #                 print(f'Downloaded {band_name}')
    #             else:
    #                 print(f'Failed to download {band_name}: HTTP Status {response.status_code}')
    #         except Exception as e:
    #             print(f'Failed to download {band_name}: {str(e)}')
                
                
                
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=len(selectedbands)) as executor:
    #         futures = {executor.submit(download_band, image.getDownloadUrl({
    #             'bands': [band],
    #             'region': aoi_ee.geometry(),
    #             'scale': scale,
    #             'format': 'GEO_TIFF'
    #         }), band): band for band in selectedbands}
    #         print('Downloading thread initiated')
    #         for future in concurrent.futures.as_completed(futures):
    #             band = futures[future]
    #             try:
    #                 future.result()  # Collect results or exceptions here if needed
    #                 urls[band] = f'./temp/{band}.tif'  # Store the downloaded file paths
    #             except Exception as e:
    #                 print(f'Failed to download {band}: {str(e)}')
    #     print('All downloads completed.')
    #     return urls
    
    
    
    def downloadBands(image, selectedbands, aoi_path, export_path, scale):
        import concurrent.futures
        import geemap
        import geedim
        import os
        from retry import retry

        aoi_ee = geemap.shp_to_ee(aoi_path)
        image_id = image.get('system:index').getInfo()
        
        export_path = f'{export_path}{image_id}/'
        
        os.makedirs(export_path, exist_ok=True)
        
        for band in selectedbands:
            band_image = image.select(band).clip(aoi_ee)
            geemap.download_ee_image(band_image,scale=scale,filename=f'{export_path}{band}.tif',dtype='float32',region=aoi_ee.geometry())
        print(f'All downloads saved to: {export_path}')
        return export_path
    
    
    def loadBands(export_path, selectedbands):
        import os
        import rasterio
        from affine import Affine
        
        bandspath = f'{export_path}'
        
        # Initialize a list to store the loaded bands
        loaded_bands = {}
    
        # List all files in the directory
        for filename in os.listdir(bandspath):
            if filename.endswith('.tif'):
                # Create the full file path
                full_filepath = os.path.join(bandspath, filename)
                
                # Open the TIFF file using rasterio
                with rasterio.open(full_filepath) as src:
                    # Read the band data and add it to the list
                    band_data = src.read(1)  # Change the band number (1) as needed
                    loaded_bands[filename.split('.')[0]]= band_data # Store the filename along with the band data
                    affine_transform = src.transform
                    crs = src.crs
         
        organized_bands = []

        # Iterate through bands_to_include and retrieve the corresponding values
        for band in selectedbands:
            if band in loaded_bands:
                organized_bands.append(loaded_bands[band]) 
        return organized_bands, affine_transform, crs
    




    def loadTrainTestData(BandData, bands_to_include, datasetpath, BandsAffineTransform, BandsCrs):
        import geopandas as gpd
        from tqdm import tqdm
        import rasterio as rio
        import pandas as pd
        import numpy as np
        from .ee_preprocessing import Preprocessing as Preprocessor

        dataset_gpd = gpd.read_file(datasetpath)
        preprocessedData = Preprocessor.preprocessShapeFiles(dataset_gpd, BandsCrs)
        # Extracting the spectral values from the sentinel image that was downloaded for the training points

        number_of_points=len(preprocessedData["fid"])
        spectrum_data=[] #This list will store the spectrum information at each training point.

        for n in tqdm(range(number_of_points)):
            point = preprocessedData.iloc[n]

            row, col = rio.transform.rowcol(BandsAffineTransform, point.geometry.x, point.geometry.y)
            spectrum_data_at_xy=[]

            for band in BandData:
                spectrum_data_at_xy.append(band[row,col])
            spectrum_data_at_xy.append(point.c_number)
            spectrum_data.append(spectrum_data_at_xy)
        spectrumdata_df = pd.DataFrame(spectrum_data) # create a new dataframe with spectrum data
        temp_spdf_cols = bands_to_include+['c_number'] #Just a temporary variable to get the column names. Nothing special
        spectrumdata_df.columns = temp_spdf_cols # set the column names for the data frame
        spectrumdata_df['index']=spectrumdata_df.index #create a column name index

        NumOfClasses=len(spectrumdata_df.c_number.unique())
        Data=[]
        Label=[]        
        for i in range(NumOfClasses):
            Data_class_i=spectrumdata_df[spectrumdata_df["c_number"]==i][bands_to_include].values
            Labels_class_i=spectrumdata_df[spectrumdata_df["c_number"]==i]["c_number"]
            Data.extend(Data_class_i)
            Label.extend(Labels_class_i)     
        Data=np.array(Data)
        Label=np.array(Label)

        return spectrumdata_df, Data, Label





    def imagePrediction(model, bandsData, classifier):
        '''Function to classify the complete image
        inputs: model- model to use to classify the image
                bandsData- the image that is to classify
                fclass- preffered class in visualisation. 
        output: predicted_Image
        '''
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
    
   
    
    
    
    
    
    # def imagePrediction(model, bandsData, classifier):
    # #     '''Function to classify the complete image
    # #     inputs: model- model to use to classify the image
    # #             bandsData- the image that is to classify
    # #             fclass- preffered class in visualisation. 
    # #     output: predicted_Image
    # #     '''
    #     from concurrent.futures import ThreadPoolExecutor
    #     import numpy as np
    #     # from tqdm import tqdm
    #     # from datetime import datetime
        
    #     predicted_LULC = []
    #     column_length = bandsData[0].shape[1]
        
    #     def classify_row(i):
    #         spectrum_data_at_row_i = [bandsData[m][i] for m in range(len(bandsData))]
    #         spectrum_data_at_row_i_T = np.transpose(np.array(spectrum_data_at_row_i))
            
    #         # if classifier == 'RF' or classifier == 'ANN':
    #         if classifier == 'ANN':
    #             predicted_class = model.predict(spectrum_data_at_row_i_T)
    #             return np.argmax(predicted_class, axis=1)
    #         else:
    #             predicted_class = np.transpose(model.predict(spectrum_data_at_row_i_T))
    #             return predicted_class

    #     # Create a ThreadPoolExecutor with the number of threads you want to use
    #     with ThreadPoolExecutor() as executor:
    #         for i in range(bandsData[0].shape[0]):
    #             predicted_LULC.append(executor.submit(classify_row, i))
    
    #     # Get the results from the futures
    #     predicted_LULC = [future.result() for future in predicted_LULC]
    #     predicted_LULC = np.array(predicted_LULC).astype(np.uint8)
    #     print(predicted_LULC)
    #     return predicted_LULC
    
    
    
    def save_Predicted_Image(algorithm_name, classified_array, export_path, BandsCrs, BandsAffineTransform):
        import rasterio as rio
        import numpy as np

        OutputFilePath_ = export_path #f"{export_path}{algorithm_name}.tif"
        with rio.open(OutputFilePath_, 'w', driver='GTiff', width=classified_array.shape[1],
                      height=classified_array.shape[0], count=1, crs=BandsCrs,
                      transform=BandsAffineTransform, dtype=np.uint8) as output:
            output.write(classified_array, 1)
        
        return 'Export Successful'


    
    def getSVMRestults(trainingData, trainingLabel, testingData, testingLabel, BandData):
        from .algorithms import Algorithms
        import joblib
        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score
        
        trainData_SVM = np.array(trainingData)
        trainLabel_SVM = trainingLabel
        testData_SVM = np.array(testingData)
        testLabel_SVM = testingLabel
        
        trainedSVM = Algorithms.fitSVM(trainData_SVM, trainLabel_SVM)
        SVM_testPredictions = trainedSVM.predict(testData_SVM)
        labels = trainedSVM.classes_
        # Convert labels to a list of strings, function classification report takes it as string only
        labels = [str(label) for label in labels]
        
        SVM_accuracy = trainedSVM.score(testData_SVM, testLabel_SVM)
        SVM_kappa = cohen_kappa_score(testLabel_SVM, SVM_testPredictions, labels=labels, weights=None, sample_weight=None)
        SVM_F1 =  f1_score(testLabel_SVM, SVM_testPredictions, average=None)
        SVM_ClassificationReport = classification_report(testLabel_SVM, SVM_testPredictions, target_names = labels)
        SVM_ConfusionMatrix = confusion_matrix(testLabel_SVM, SVM_testPredictions, labels=trainedSVM.classes_)

        # SVM_ClassifiedImage = Utilities.imagePrediction(trainedSVM, BandData, 'SVM')
        
        
        return {
            'Classifier':'SVM',
            'Accuracy':SVM_accuracy, 
            'Kappa': SVM_kappa, 
            'F1': SVM_F1,
            'TestData':testingData, 'TestPredictions': SVM_testPredictions, 'SVMLabels': labels, 'Model': trainedSVM,
            'ClassificationReport': SVM_ClassificationReport, 'ConfusionMatrix': SVM_ConfusionMatrix,
                }

    def getRFResults(trainingData, trainingLabel, testingData, testingLabel, BandData):
        from .algorithms import Algorithms
        import joblib
        import numpy as np
        from tensorflow import keras
        from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, f1_score
        
        trainData_RF = np.array(trainingData)
        trainLabel_RF = trainingLabel
        testData_RF = np.array(testingData)
        testLabel_RF = testingLabel
        
        trainedRF = Algorithms.fitRF(trainData_RF, trainLabel_RF)
        RF_testPredictions = trainedRF.predict(testData_RF)
        print('Test predictions')
        print(RF_testPredictions)
        labels = trainedRF.classes_
        print('labels---', labels)
        labels = [str(label) for label in labels]
        
        RF_accuracy = trainedRF.score(testData_RF, testLabel_RF)
        print('Accuracy', RF_accuracy)
        RF_kappa = cohen_kappa_score(testLabel_RF, RF_testPredictions, labels=labels, weights=None, sample_weight=None)
        print('kappa', RF_kappa)
        RF_F1 =  f1_score(testLabel_RF, RF_testPredictions, average=None)
        print('F1', RF_F1)
        RF_ClassificationReport = classification_report(testLabel_RF, RF_testPredictions, target_names = labels)
        print(RF_ClassificationReport)
        RF_ConfusionMatrix = confusion_matrix(testLabel_RF, RF_testPredictions, labels=labels)
        
        return {
            'Classifier':'RF',
            'Accuracy':RF_accuracy, 
            'Kappa': RF_kappa, 
            'F1': RF_F1,
            'TestData':testingData, 'TestPredictions': RF_testPredictions, 'SVMLabels': labels, 'Model': trainedRF,
            'ClassificationReport': RF_ClassificationReport, 'ConfusionMatrix': RF_ConfusionMatrix,
            }
    
    
    def getANNRestults(trainingData, trainingLabel, testingData, testingLabel, BandData):
        from .algorithms import Algorithms
        import joblib
        import numpy as np
        from tensorflow import keras
        
        trainData_ANN = np.array(trainingData)
        trainLabel_ANN = keras.utils.to_categorical(trainingLabel)
        testData_ANN = np.array(testingData)
        testLabel_ANN = keras.utils.to_categorical(testingLabel)
        
        trainedANN = Algorithms.fitANN(trainData_ANN, trainLabel_ANN, testData_ANN, testLabel_ANN)
        ANN_accuracy = trainedANN.score(testData_ANN, testLabel_ANN)
        
        ANN_ClassifiedImage = Utilities.imagePrediction(trainedANN, BandData, 'ANN')
        
        
        return ANN_accuracy, ANN_ClassifiedImage
    
        
        # def classifyUsingTrainedSVM(BandData, BandsAffineTransform, BandsCrs, export_path):
        #     import joblib
        #     import rasterio as rio
            
        #     trainedSVMNoIndices = joblib.load('./libfiles/trainedmodels/SVM_WithNoIndicesGrid_BestModel0944047619047619.joblib')
        #     trainedSVM4Indices = joblib.load('./libfiles/trainedmodels/SVM_With4IndicesGrid_BestModel09380952380952381.joblib')
            
            
        #     # imgPrediction_SVM_NoIndices = imagePrediction(trainedSVMNoIndices, BandData, 'SVM')
        #     imgPrediction_SVM_4Indices = imagePrediction(trainedSVM4Indices, BandData, 'SVM')
            
        #     output_path_no_NoIndices = f'{export_path}SVM_NoIndices_classified'
        #     output_path_no_4Indices = f'{export_path}SVM_NoIndices_classified'
            
        #     with rio.open(output_path, 'w', driver='GTiff', width=rfimage.shape[1],
        #                   height=rfimage.shape[0], count=1, crs=crs_rio,
        #                   transform=transform_rio, dtype=np.uint8) as output:
        #         output.write(rfimage, 1)
        #     Image.fromarray(255*(rfimage==2).astype(np.uint8)).show()
            
        #     return imgPrediction_SVM_4Indices
            
            
    def normalize(BandData):
        import numpy as np
        for index, array in enumerate(BandData):
            array_min, array_max = np.nanmin(array), np.nanmax(array)  
            normalized = ((array - array_min) / (array_max - array_min))
            BandData[index]=normalized
        
        return BandData
     
    # Define a function to save a single-band raster to a GeoTIFF file
    def save_geotiff(data, crs, transform, output_path):
        import rasterio as rio
        height, width = data.shape
        metadata = {
            'driver': 'GTiff',
            'count': 1,
            'dtype': data.dtype,
            'width': width,
            'height': height,
            'crs': crs,
            'transform': transform,
        }

        with rio.open(output_path, 'w', **metadata) as dst:
            dst.write(data, 1)

        
            
        
        
