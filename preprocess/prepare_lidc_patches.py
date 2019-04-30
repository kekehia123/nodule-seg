# Run this file to prepare 64*64 patches of lidc.

# In this notebook, we want to save consecutive three slices as one png.
# We need to be careful with the HU crop value.
# Set where to save the prepared data
# You should make 3 subdirs here: scans, patches, masks 

config = {}
config['savepath'] = '/home/shenxk/Documents/nodule_seg/prep_result/previous'
os.chdir(config['savepath'])
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import time

import pylidc as pl
from pylidc.Annotation import feature_names as fnames
from scipy.misc import imsave as imsave
import PIL.Image as Image
import matplotlib.image as mpimg

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import scipy.ndimage

from preprocessing import full_prep
from preprocessing import step1
from preprocessing import preprocessing_lib as prep_lib

with open('error.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['short_name','cluster'])
    
with open('nodule_size.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['short_name','cluster','size'])
    
with open('scan_size.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['short_name','size'])
    
# Generate mask from the four annotations for each nodule. 
# Only nodules with four annotations available are included. 
# Only voxels marked by three or four radiologists are regarded as nodule voxels.
all_scans = pl.query(pl.Scan)
config['patch_size'] = 64
savepath = config['savepath']
start = time.clock()
num_scans = all_scans.count()
#num = 2
all_anns = pl.query(pl.Annotation).join(pl.Scan)
for scan_id in range(num_scans):
    anns = all_anns.filter(pl.Annotation.scan_id==scan_id)
    scan = all_scans[scan_id]
    shortname_scan = '0'*(4-len(str(scan_id)))+str(scan_id)
    
    try:
        path = scan.get_path_to_dicom_files()
    except AssertionError:
        print(shortname_scan, 'does not exist')
        continue
    else:
        images = scan.load_all_dicom_images()
        first_image = images[0].pixel_array
        [width, height] = first_image.shape
        num_slices = len(images)
        images_array = np.zeros((num_slices, width, height))
        for i in range(num_slices):
            images_array[i,:,:] = images[i].pixel_array
        image_tran = images_array.transpose((1,2,0))
        slice_thickness = scan.slice_thickness
        
        with open('scan_size.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([shortname_scan, image_tran.shape])
        print(shortname_scan)

        anns_clustered = scan.cluster_annotations()
        
        for cluster in range(len(anns_clustered)):
            if len(anns_clustered[cluster]) == 4:
                # Get the min slice of the bbox
                min_slice = 1000
                max_slice = 0
                contours = [0,0,0,0]
                fnames = [0,0,0,0]
                index_of_contour = {}
                for i in range(4):
                    contours[i] = sorted(anns_clustered[cluster][i].contours, key=lambda c: c.image_z_position)
                    fnames[i] = anns_clustered[cluster][i].scan.sorted_dicom_file_names.split(',')
                    index_of_contour[i] = [fnames[i].index(c.dicom_file_name) for c in contours[i]]
                    min_slice = min(min(index_of_contour[i]),min_slice)
                    max_slice = max(max(index_of_contour[i]),max_slice)
                #print(index_of_contour[0][0]:index_of_contour[0][-1]+1)
                
                mask0, bbox0 = anns_clustered[cluster][0].get_boolean_mask(return_bbox=True)
                mask1, bbox1 = anns_clustered[cluster][1].get_boolean_mask(return_bbox=True)
                mask2, bbox2 = anns_clustered[cluster][2].get_boolean_mask(return_bbox=True)
                mask3, bbox3 = anns_clustered[cluster][3].get_boolean_mask(return_bbox=True)
                bbox_min = np.zeros(3)
                bbox_min[0] = np.min([bbox0[0,0], bbox1[0,0], bbox2[0,0], bbox3[0,0]])
                bbox_min[1] = np.min([bbox0[1,0], bbox1[1,0], bbox2[1,0], bbox3[1,0]])
                bbox_min[2] = np.min([bbox0[2,0], bbox1[2,0], bbox2[2,0], bbox3[2,0]])
                bbox_max = np.zeros(3)
                bbox_max[0] = np.max([bbox0[0,1], bbox1[0,1], bbox2[0,1], bbox3[0,1]])
                bbox_max[1] = np.max([bbox0[1,1], bbox1[1,1], bbox2[1,1], bbox3[1,1]])
                bbox_max[2] = np.max([bbox0[2,1], bbox1[2,1], bbox2[2,1], bbox3[2,1]])

                mask_new = np.zeros((4,int(bbox_max[0] - bbox_min[0]+1), int(bbox_max[1] - bbox_min[1]+1),
                                     max_slice-min_slice+1))

                margin0 = bbox0[:,0] - bbox_min
                margin1 = bbox1[:,0] - bbox_min
                margin2 = bbox2[:,0] - bbox_min
                margin3 = bbox3[:,0] - bbox_min
                margin0 = np.asarray(margin0, dtype=int)
                margin1 = np.asarray(margin1, dtype=int)
                margin2 = np.asarray(margin2, dtype=int)
                margin3 = np.asarray(margin3, dtype=int)
                #z_margin0 = int(margin0[2]/slice_thickness)
                #z_margin1 = int(margin1[2]/slice_thickness)
                #z_margin2 = int(margin2[2]/slice_thickness)
                #z_margin3 = int(margin3[2]/slice_thickness)
                
                # Get the coordinate of the lower bound in the original scan. 
                lower_bound = [int(bbox_min[0]),int(bbox_min[1]),min_slice]
                #print(bbox0)
                #print(mask0.shape)
                #print(mask_new[0, margin0[0]:int(margin0[0]+(bbox0[0,1]-bbox0[0,0])+1), margin0[1]:int(margin0[1]+(bbox0[1,1]-bbox0[1,0])+1),
                #             index_of_contour[0][0]-min_slice:index_of_contour[0][-1]-min_slice+1].shape)
                #print(index_of_contour[0])
                #print(min_slice)
                #print(mask_new.shape)
                
                #zs = np.unique([c.image_z_position for c in anns_clustered[cluster][3].contours])
                #z_to_index = dict(zip(zs,range(len(zs))))
                try:
                    mask_new[0, margin0[0]:int(margin0[0]+(bbox0[0,1]-bbox0[0,0])+1), margin0[1]:int(margin0[1]+(bbox0[1,1]-bbox0[1,0])+1),
                             index_of_contour[0][0]-min_slice:index_of_contour[0][-1]-min_slice+1] = mask0
                    mask_new[1, margin1[0]:int(margin1[0]+(bbox1[0,1]-bbox1[0,0])+1), margin1[1]:int(margin1[1]+(bbox1[1,1]-bbox1[1,0])+1),
                             index_of_contour[1][0]-min_slice:index_of_contour[1][-1]-min_slice+1] = mask1
                    mask_new[2, margin2[0]:int(margin2[0]+(bbox2[0,1]-bbox2[0,0])+1), margin2[1]:int(margin2[1]+(bbox2[1,1]-bbox2[1,0])+1),
                             index_of_contour[2][0]-min_slice:index_of_contour[2][-1]-min_slice+1] = mask2
                    mask_new[3, margin3[0]:int(margin3[0]+(bbox3[0,1]-bbox3[0,0])+1), margin3[1]:int(margin3[1]+(bbox3[1,1]-bbox3[1,0])+1),
                             index_of_contour[3][0]-min_slice:index_of_contour[3][-1]-min_slice+1] = mask3
                except ValueError:
                    print(shortname_scan, 'error')
                    with open('error.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([shortname_scan, cluster])
                    continue
                else:
                    [p_mask, width_mask, height_mask, z_mask] = mask_new.shape
                    mask_sum = np.zeros(mask_new.shape[1:])
                    mask_overlap = np.zeros(mask_new.shape[1:])
                    with open('nodule_size.csv', 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([shortname_scan, cluster, mask_overlap.shape])
                    #print(mask_overlap.shape)

                    for i in range(width_mask):
                        for j in range(height_mask):
                            for k in range(z_mask):
                                mask_sum[i,j,k] = np.sum(mask_new[:,i,j,k])
                                if mask_sum[i,j,k] >= 3:
                                    mask_overlap[i,j,k] = 1



                    # Define the size of the bounding box
                    box_size = config['patch_size']
                    # Cut the bounding box according to the size of the overlapped annotation area.
                    mask_where = np.argwhere(mask_overlap == 1)
                    box_new_min = np.min(mask_where, axis = 0)
                    box_new_max = np.max(mask_where, axis = 0)
                    mask_overlap_cut = mask_overlap[box_new_min[0]:box_new_max[0]+1, box_new_min[1]:box_new_max[1]+1,
                                                    box_new_min[2]:box_new_max[2]+1]
                    [width_cut, height_cut, z_cut] = mask_overlap_cut.shape

                    # Pad the mask to get the desired size
                    num_pad_width0 = int(np.floor((box_size - width_cut)/2))
                    num_pad_width1 = int(box_size - width_cut - num_pad_width0)
                    num_pad_height0 = int(np.floor((box_size - height_cut)/2))
                    num_pad_height1 = int(box_size - height_cut - num_pad_height0)
                    mask_pad = np.pad(mask_overlap_cut, ((num_pad_width0, num_pad_width1),(num_pad_height0, num_pad_height1),(0,0)), 
                                      'constant', constant_values = ((0,0),(0,0),(0,0)))

                    # Save the padded masks. 
                    # Each consecutive three slices are saved in one npy, resulted in num_slice-2 npy files.
                    cluster_id = '0'*(2-len(str(cluster)))+str(cluster)
                    
                    mask_pad = np.asarray(mask_pad, dtype=np.uint8)

                    
                    for i in range(z_cut-2):
                        #print(z_cut)
                        #mask_pad_save = mask_pad[:,:,i:i+3]
                        #print(mask_pad_save.shape)
                        #np.save(os.path.join(savepath, 'masks', 'mask'+shortname_scan+cluster_id+'0'*(2-len(str(i)))+str(i)),
                        #        mask_pad[:,:,i:i+3])
                        im = Image.fromarray(mask_pad[:,:,i+1])
                        im.save(os.path.join(savepath, 'masks', 'mask'+shortname_scan+cluster_id+'0'*(2-len(str(i)))+str(i))+'.png')
                        #imsave(os.path.join(savepath, 'masks', 'mask'+shortname_scan+cluster_id+'0'*(2-len(str(i)))+str(i)+'.png'),
                        #        mask_pad_save)

                    # Get the coordinate of padded mask's lower bound in the original scan.
                    lower_bound = lower_bound + box_new_min - [num_pad_width0, num_pad_height0, 0]

                    # Get the corresponding ct patch
                    ct_patch = image_tran[lower_bound[0]:lower_bound[0]+box_size, lower_bound[1]:lower_bound[1]+box_size, 
                                            lower_bound[2]:lower_bound[2]+z_cut]
                    #print(np.min(ct_patch),np.max(ct_patch))
                    
                    # Clip the patch by the boundary of [-1200, 600], then normalize to [0, 255]
                    ct_patch[ct_patch > 600] = 600
                    ct_patch[ct_patch < -1200] = -1200
                    ct_patch = (ct_patch + 1200) / 1800 * 255  
                    ct_patch = np.asarray(ct_patch, dtype=np.uint8)
                    #print(np.min(ct_patch),np.max(ct_patch))
                    for i in range(z_cut-2):
                        #np.save(os.path.join(savepath, 'patches','ct_patch'+shortname_scan+cluster_id+'0'*(2-len(str(i)))+str(i)),
                        #       ct_patch[:,:,i:i+3])
                        im = Image.fromarray(ct_patch[:,:,i:i+3])
                        #im.save(os.path.join(savepath, 'patches','ct_patch'+shortname_scan+cluster_id+'0'*(2-len(str(i)))+str(i))+'.png')
                        imsave(os.path.join(savepath, 'patches','ct_patch'+shortname_scan+cluster_id+'0'*(2-len(str(i)))+str(i))+'.png',
                                ct_patch[:,:,i:i+3])

elapsed = (time.clock() - start)
print('Time used:', elapsed)