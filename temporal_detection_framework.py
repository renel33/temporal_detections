import rasterio
import glob
import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt


def show_image(array):
    plt.imshow(array)
    plt.show()
    plt.close()


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
    
    else:
        detection_tensor = []
        for file in files:
            with rasterio.open(file) as src:
                rst = src.read(1)
                if rst.shape == shape:
                    detection_tensor.append(rst)
        return np.asarray(detection_tensor, dtype=np.float32)
    

def average_tensor(tensor, step_size=10, thresh=False):
    if type(tensor)==list:
        date_tensors = []
        for ten in tensor:
            tensors = []
            iend = ten.shape[0] - step_size
            jend = ten.shape[0]
            jstart = step_size
            
            if step_size >= ten.shape[0]:
                avg_tensor = np.mean(ten[0:jend], axis=0)
                if thresh == True:
                    avg_tensor = threshold_tensor(avg_tensor)
                tensors.append(avg_tensor)
                date_tensors.append(np.asarray(tensors, dtype=np.float32))
            
            else:
                for i, j in zip(range(0, iend, step_size), range(jstart, jend, step_size)):
                    avg_tensor = np.mean(ten[i:j], axis=0)
                    if thresh == True:
                        avg_tensor = threshold_tensor(avg_tensor)
                    tensors.append(avg_tensor)
                date_tensors.append(np.asarray(tensors, dtype=np.float32))
        return date_tensors
    else:   
        tensors = []
        iend = tensor.shape[0] - step_size
        jend = tensor.shape[0]
        
        for i, j in zip(range(0, iend, step_size), range(step_size, jend, step_size)):
            avg_tensor = np.mean(tensor[i:j], axis=0)
            if thresh == True:
                avg_tensor = threshold_tensor(avg_tensor)
            tensors.append(avg_tensor)
        return np.asarray(tensors, dtype=np.float32)

def median_tensor(tensor):
    if type(tensor)==list:
        date_tensors = []
        for ten in tensor:
            med_tensor = np.median(ten, axis=0)
            date_tensors.append(np.asarray(med_tensor, dtype=np.float32))
        return date_tensors
    else:   
        tensor = np.mean(tensor, axis=0)
        return np.asarray(tensor, dtype=np.float32)

def tensor_stats(tensor, stats, threshold=None):
    if type(tensor) == list:
        list_of_stats_arrays = []
        for ten in tensor:
            stats_array = []
            if "max" in stats:
                max_thr_ten = np.max(np.asarray(ten, dtype=np.float32), axis=0)
                if threshold is not None:
                    max_thr_ten[max_thr_ten >= threshold] = 1
                    max_thr_ten[max_thr_ten < threshold] = 0
                stats_array.append(max_thr_ten)
            if "mean" in stats:
                avg_thr_ten = np.mean(np.asarray(ten, dtype=np.float32), axis=0)
                if threshold is not None:
                    avg_thr_ten[avg_thr_ten >= threshold] = 1
                    avg_thr_ten[avg_thr_ten < threshold] = 0
                stats_array.append(avg_thr_ten)
            if "median" in stats:
                med_thr_ten = np.median(np.asarray(ten, dtype=np.float32), axis=0)
                if threshold is not None:
                    med_thr_ten[med_thr_ten >= threshold] = 1
                    med_thr_ten[med_thr_ten < threshold] = 0
                stats_array.append(med_thr_ten)
            list_of_stats_arrays.append(stats_array)
        return list_of_stats_arrays
    else:
        stats_array = []
        if "max" in stats:
            max_thr_ten = np.max(np.asarray(tensor, dtype=np.float32), axis=0)
            if threshold is not None:
                    max_thr_ten[max_thr_ten >= threshold] = 1
                    max_thr_ten[max_thr_ten < threshold] = 0
            stats_array.append(max_thr_ten)
        if "mean" in stats:
            avg_thr_ten = np.mean(np.asarray(tensor, dtype=np.float32), axis=0)
            if threshold is not None:
                    avg_thr_ten[avg_thr_ten >= threshold] = 1
                    avg_thr_ten[avg_thr_ten < threshold] = 0
            stats_array.append(avg_thr_ten)
        if "median" in stats:
            med_thr_ten = np.median(np.asarray(tensor, dtype=np.float32), axis=0)
            if threshold is not None:
                    med_thr_ten[med_thr_ten >= threshold] = 1
                    med_thr_ten[med_thr_ten < threshold] = 0
            stats_array.append(med_thr_ten)
        return(stats_array)


def write_stat_rasters(tensors, in_file, out_file, year_list):
    files = glob.glob(f'{in_file}/*.tif')
    with rasterio.open(files[0]) as src:
                    out_meta = src.meta
                    out_meta.update({"driver": "GTiff",
                                "height": assert_shape(in_file)[0],
                                "width": assert_shape(in_file)[1]})
    if type(tensors[0]) == list:
        for object, date in zip(tensors, year_list):
        
            if not os.path.exists(out_file):
                os.makedirs(out_file)

            with rasterio.open(f'{out_file}/max_detections_{date}.tif', "w", **out_meta) as dest:
                dest.write(object[0], 1)

            with rasterio.open(f'{out_file}/mean_detections_{date}.tif', "w", **out_meta) as dest:
                dest.write(object[1], 1)

            with rasterio.open(f'{out_file}/median_detections_{date}.tif', "w", **out_meta) as dest:
                dest.write(object[2], 1)
    
    else:
        if not os.path.exists(out_file):
                os.makedirs(out_file)

        with rasterio.open(f'{out_file}/max_detections.tif', "w", **out_meta) as dest:
            dest.write(object[0], 1)

        with rasterio.open(f'{out_file}/mean_detections.tif', "w", **out_meta) as dest:
            dest.write(object[1], 1)
            
        with rasterio.open(f'{out_file}/median_detections.tif', "w", **out_meta) as dest:
            dest.write(object[2], 1)


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
                
                with rasterio.open(f'{out_file}/averaged_tensors_{count}.tif', "w", **out_meta) as dest:
                    dest.write(tensor, 1)
                
                count += 1


def show_tensors(tensors):
    for object in tensors:
        if type(object) == list:
            for tensor in object:
                show_image(tensor)
        else:
            show_image(object)


def absolute_average(tensors, thresh=False):
    tensors_out = []
    for tensor in tensors:
        if tensor.shape[0] == 0:
            continue
        if thresh == True:
            tensor = threshold_tensor(tensor)
        tensors_out.append(np.mean(tensor, axis=0))
    return np.asarray(tensors_out, np.float32)


def tensor_difference(tensors, year_list):
    out_tensors = []
    for index in range(0,len(tensors),1):
        if index + 1 == len(tensors):
            break
        tensor = tensors[index]
        next_tensor = tensors[index + 1]
        year = year_list[index]
        next_year = year_list[index + 1]
        diff = tensor - next_tensor
        out_tensors.append(diff)
    return np.asarray(out_tensors, np.float32)

def threshold_tensor(tensor):
    tensor[tensor < 1] = 0
    return tensor
        

if __name__ == "__main__":
    in_file = "/home/rene1337/RSCPH/PlanetTimeseriesTest/output_predictions/20230301-1203_test/rasters"
    out_file = "/home/rene1337/RSCPH/PlanetTimeseriesTest/multi_years/"
    years = [str(year) for year in range(2018, 2023, 1)]
    
    tensors = load_tensor(in_file, years)
    
    med_tensors = median_tensor(tensors)
    
    average_tensors = average_tensor(tensors, 10, True)
    
    absolute_average_tensors = absolute_average(average_tensors, True)
    
    #med_diff = tensor_difference(med_tensors, years)
    #avg_diff = tensor_difference(average_tensors, years)
    abs_avg_diff = tensor_difference(absolute_average_tensors, years)

    
    #write_rasters(tensors, in_file, "/home/rene1337/RSCPH/PlanetTimeseriesTest//", years)
    #write_rasters(med_diff, in_file, "/home/rene1337/RSCPH/PlanetTimeseriesTest/median_diff_years/", [str(year) for year in range(2018, 2022, 1)])
    #write_rasters(avg_diff, in_file, "/home/rene1337/RSCPH/PlanetTimeseriesTest/avg_diff_years/", [str(year) for year in range(2018, 2022, 1)])
    write_rasters(abs_avg_diff, in_file, "/home/rene1337/RSCPH/PlanetTimeseriesTest/abs_avg_diff_years/", [str(year) for year in range(2018, 2022, 1)])
    write_rasters(average_tensors, in_file, "/home/rene1337/RSCPH/PlanetTimeseriesTest/average_thresh_years/", years)
    
    
    
        
    
    #show_tensors(diff)
    #[print(tensor.shape) for tensor in tensors]
    # tensors = tensor_stats(tensors, stats=["max", "mean", "median"], threshold=0.7)
    #write_stat_rasters(tensors, in_file, out_file, years)
    
    
    



