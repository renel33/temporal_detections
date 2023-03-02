import rasterio
import glob
import numpy as np
import os
from osgeo import gdal
import matplotlib.pyplot as plt

in_file = "/home/rene1337/RSCPH/MultiTimeStepPlanetTest/output_predictions/20230222-1226_test/rasters"
out_file = "/home/rene1337/RSCPH/MultiTimeStepPlanetTest/conc_dets/"
files = sorted(glob.glob(f"{in_file}/*.tif"))


def fig_show(array):
    plt.imshow(array)
    plt.show()
    plt.close()


detection_tensor = []
for file in files:
    with rasterio.open(file) as src:
        rst = src.read(1)
        if rst.shape == (609, 616):
            detection_tensor.append(rst) 
dtensor = np.asarray(detection_tensor, dtype=np.float32)


thresh_tensors = []
for i, j in zip(range(0, 50, 10), range(10, 56, 10)):
    avg_dtensor = np.mean(dtensor[i:j], axis=0)
    avg_dtensor[avg_dtensor < 0.5] = 0
    thresh_tensors.append(avg_dtensor)


max_thr_ten = np.max(np.asarray(thresh_tensors, dtype=np.float32), axis=0)
avg_thr_ten = np.mean(np.asarray(thresh_tensors, dtype=np.float32), axis=0)
med_thr_ten = np.median(np.asarray(thresh_tensors, dtype=np.float32), axis=0)


max_thr_ten[max_thr_ten == 0.5] = 1
avg_thr_ten[avg_thr_ten >= 0.1] = 1
med_thr_ten[med_thr_ten == 0.5] = 1


with rasterio.open(files[0]) as src:
    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                    "height": max_thr_ten.shape[0],
                    "width": max_thr_ten.shape[1]})
    

if not os.path.exists(out_file):
    os.makedirs(out_file)


with rasterio.open(f'{out_file}/max_detections.tif', "w", **out_meta) as dest:
    dest.write(max_thr_ten, 1)


with rasterio.open(f'{out_file}/mean_detections.tif', "w", **out_meta) as dest:
    dest.write(avg_thr_ten, 1)


with rasterio.open(f'{out_file}/median_detections.tif', "w", **out_meta) as dest:
    dest.write(med_thr_ten, 1)


