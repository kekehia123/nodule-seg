import matplotlib.pyplot as plt
import numpy as np
import os
import csv

import pylidc as pl
from pylidc.Annotation import feature_names as fnames
from scipy.misc import imsave 
import PIL.Image as Image
import matplotlib.image as mpimg

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import scipy.ndimage


def normalize(img, low, high):
    ret = (np.clip(img, a_min=low, a_max=high))
    ret = (ret - low) / (high - low)
    ret = (ret * 255).astype(np.uint8)
    return ret

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    for slice_number in range(len(slices)):       
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)         
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

all_scans = pl.query(pl.Scan)
num_scans = all_scans.count() 

def save_slices(save_path, id_range = [0, num_scans], norm_range = np.array([[-1000, 200], [-250, 200], [-1000, -745]])):
    # Save slices as png with 3 channels
    
    # Create a table to save the corresponding origin path of the saved npys.
    with open('names.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['short_name','origin_path'])

    with open('miss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['short_name'])

    #Save each scan as a series of png file.
    for id in range(id_range[0], id_range[1]):
        shortname = '0'*(4-len(str(id)))+str(id)
        try:
            path = all_scans[id].get_path_to_dicom_files()
        except AssertionError:
            with open('miss.csv', 'a') as f:      
                writer = csv.writer(f)
                writer.writerow([shortname])
            continue
        else:
            images = all_scans[id].load_all_dicom_images()
            
            first_image = images[0].pixel_array
            [width, height] = first_image.shape
            num_slices = len(images)
            
            images_hu = get_pixels_hu(images)
            
            image_array = np.zeros((width, height, 3))

            for i in range(num_slices):
                slice_array = images_hu[i]
                # Save 3 channel normalized images
                image_array[:,:,0] = normalize(slice_array, norm_range[0, 0], norm_range[0, 1])
                image_array[:,:,1] = normalize(slice_array, norm_range[1, 0], norm_range[1, 1])
                image_array[:,:,2] = normalize(slice_array, norm_range[2, 0], norm_range[2, 1])
                
                image_name = os.path.join(save_path, shortname+'0'*(3-len(str(i)))+str(i)+'.png')
                imsave(image_name, image_array)

            print(shortname)
            print(num_slices)
            
            with open('names.csv', 'a') as f:      
                writer = csv.writer(f)
                writer.writerow([shortname, path])
                               
save_slices('/home/shenxk/data/LungCancerData/LIDC_prep/slices', id_range = [0, num_scans])