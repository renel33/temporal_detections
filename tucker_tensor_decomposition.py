import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sktensor import dtensor
from sktensor import tucker
from sktensor.core import ttm
import rasterio
import glob
import os

def assert_shape(in_file):
    
    files = sorted(glob.glob(f"{in_file}/*clip.tif"))
    start_shape = (0,0)
    for file in files:
        with rasterio.open(file) as src:
            rst = src.read(1)
            if rst.shape > start_shape:
                start_shape = rst.shape
    return start_shape


def load_tensor(in_file, year_list=None, shape=None):
    
    if shape is None:
        shape = assert_shape(in_file)
    
    files = sorted(glob.glob(f"{in_file}/*clip.tif"))
    if year_list is not None:
        array = []
        for year in year_list:
            detection_tensor = []
            for file in files:
                if year in file:
                    with rasterio.open(file) as src:
                        rst = src.read(1)
                        if rst.shape == shape:
                            detection_tensor.append(rst)
                            
            dtensor = np.asarray(detection_tensor, dtype=np.float32)
            array.append(dtensor)
        
        return array


def decompose(tensors, years):
    out_array = []
    for tensor in tensors:
        X = dtensor(np.array(tensor))
        # Only 1 and 2 are valid inputs when 3 imagepip s are used, 1 means 'more' shadow/sun is removed while 2 will keep some
        # you should inspect the results to pick a desired value
        R = 2
        # Q and P should aproximetly keep the same aspect ratio as a original images - IMPORTANT
        # The higher the value the more data is preserved during the decomposition process
        # Mostly likely it should get better while increasing a number till it reaches a point of saturation
        # where slightly misaligned pixels will introduce noise grater then benefits of keeping data about low variance
        # simillarly as PCA does ;)
        # That is another parameter dependent on particular image and alignment precision,
        # you should try a few different values and compare the model performance on it.
        # It should be possible to find common values for all of our data as I asume alignment performance, sieze etc are quite consistant
        Q = tensor.shape[1]
        P = tensor.shape[2]

        # Parameters to be further cusstomized if needed
        #I think it was designed with smaller matriceses then we have in mind so lilely needs to be reised, especially when high values of Q and P are used
        #__DEF_MAXITER = 500
        #__DEF_INIT = 'nvecs'
        #__DEF_CONV = 1e-7
        G, U = tucker.hooi(X, [R, Q, P], init='nvecs')

        print(G.shape)
        print(len(U))
        print(len(U[0]))
        print(len(U[1]))
        print(len(U[2]))
        #print(G)
        #print(U)

        # Reverse transformation to obtain cloud free image
        # Note that that method will result with all input images to be modified, therfore all of them should be used insted of original ones(keeping some hand picked 'cloud free images' in the data set for further ML is not a good idea)
        A = ttm(G, U)
        print(A.shape)
        for out_tensor in A:
            out_array.append(out_tensor)
            
    return A

def write_rasters(tensors, in_file, out_file, year_list):
    files = glob.glob(f'{in_file}/*.tif')
    with rasterio.open(files[0]) as src:
                    out_meta = src.meta
                    out_meta.update({"driver": "GTiff",
                                "height": assert_shape(in_file)[0],
                                "width": assert_shape(in_file)[1]})
                    
    if type(tensors) == list:
        for object, date in zip(tensors, year_list):
            count = 1
            for tensor in object:
                if not os.path.exists(out_file):
                    os.makedirs(out_file)

                with rasterio.open(f'{out_file}/averaged_tensors_{date}_{count}.tif', "w", **out_meta) as dest:
                    dest.write(tensor, 1)
                
                count += 1
    else:
        count = 1
        for tensor in tensors:
                if not os.path.exists(out_file):
                    os.makedirs(out_file)

                with rasterio.open(f'{out_file}/averaged_tensors_{count}.tif', "w", **out_meta) as dest:
                    dest.write(tensor, 1)
                
                count += 1
                
if __name__ == "__main__":
    
    in_file = "/home/rene1337/RSCPH/PlanetTimeseriesTest/output_predictions/20230301-1203_test/rasters"
    out_file = "/home/rene1337/RSCPH/PlanetTimeseriesTest/multi_years/"
    years = [str(year) for year in range(2018, 2023, 1)]
    
    tensors = load_tensor(in_file, years)
    
    decomposed = decompose(tensors, years)