import numpy as np
import os
import csv

import pylidc as pl
from pylidc.Annotation import feature_names as fnames
from scipy.misc import imsave as imsave
import PIL.Image as Image
import matplotlib.image as mpimg

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import scipy.ndimage

config = {}
# In this notebook, we want to save consecutive three slices as one png.
# We need to be careful with the HU crop value.
# Set where to save the prepared data
# You should make 3 subdirs here: scans, patches, masks 
config['savepath'] = '/home/shenxk/data/LungCancerData/LIDC_scans'
os.chdir(config['savepath'])


all_scans = pl.query(pl.Scan)
savepath = config['savepath']
#num = 2
num_scans = all_scans.count()

#Create a table to save the corresponding origin path of the saved npys.
with open('names.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['short_name','origin_path'])

with open('miss.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['short_name'])
    
#Save each scan as an npy file.
for id in range(num_scans):
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
        #print(num_slices)
        images_array = np.zeros((num_slices, width, height))
        
        for i in range(num_slices):
            images_array[i,:,:] = images[i].pixel_array
        print(shortname)
        #print(np.min(images_array), np.max(images_array))
        
        path = all_scans[id].get_path_to_dicom_files()
        np.save(os.path.join(savepath, 'scans', shortname+'.npy'), images_array)
        #imsave(os.path.join(savepath, 'scans', shortname+'.png'),images_array)

        with open('names.csv', 'a') as f:      
            #fieldnames = ['short_name', 'origin_path']
            #writer = csv.DictWriter(f, fieldnames=fieldnames)
            #writer.writeheader()
            writer = csv.writer(f)
            writer.writerow([shortname, path])