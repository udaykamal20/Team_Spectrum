#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script with helper functions to generate image and binary masks from dicom files and save them to the parent directory.
The input path must have the same structure as the provided dataset

"""
     
import pydicom as dicom
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import operator
import warnings
import glob

##function to get z axis value for all slices
def get_all_z(path):
    sorted_slices = slice_order(path)
    all_z = [p[1] for p in sorted_slices]
    return sorted_slices,all_z


def decide_validation_or_test(path):
    """
    Decide whether to generate binary mask of csv files by detecting the presence of RTStruct files.
    
    Inputs:
            path (str): path of the the directory that has patient folders to be tested
    Return:
        A binary decision. True means validation data i.e. folder has RTStruct label, False means test data
    """
    # handle `\\` missing
    if path[-1] != '\\': path += '\\'
    # get all .dcm  file
    fpaths = glob.glob(path + '*\\*\\*\\*.dcm')

    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            return 1

    return 0


    
def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence 
    inside dicom data structure.
    
    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `\\` missing
    if path[-1] != '\\': path += '\\'
    # get .dcm contour file
    fpaths = glob.glob(path + '*\\*\\*.dcm')
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = dicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath
            n += 1
    if n > 1: warnings.warn("There are multiple contour files, returning the last one!")
    if contour_file is None: print("No contour file found in directory")
    return contour_file


def get_roi_names(contour_data):
    """
    This function will return the names of different contour data, 
    e.g. different contours from different experts and returns the name of each.
    Inputs:
        contour_data (dicom.dataset.FileDataset): contour dataset, read by dicom.read_file
    Returns:
        roi_seq_names (list): names of the 
    """
    roi_seq_names = [roi_seq.ROIName for roi_seq in list(contour_data.StructureSetROISequence)]
    return roi_seq_names
    
def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `\\` missing
    if path[-1] != '\\': path += '\\'

    slices = []
    fpaths = glob.glob(path + '*\\DICOM\\1*.dcm')
    for s in fpaths:
        try:
            f = dicom.read_file(s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
        except:
            continue

    slice_dict = {fpaths[i]: s.ImagePositionPatient[-1] for i,s in enumerate(slices)}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
#    z = [p[1] for p in ordered_slices]
    return ordered_slices



def get_data(path):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_dict (dict): dictionary created by get_contour_dict
    """
    all_slice_path, all_z = get_all_z(path)
    images = []
    contours = []
    # handle `\\` missing
    if path[-1] != '\\': path += '\\'
    # get contour file
    contour_file = get_contour_file(path)
    # get slice orders
    ordered_slices = slice_order(path)
    # get contour dict
    contour_dict = get_contour_dict(contour_file, path)
    
    for i in range(len(ordered_slices)):
    
#    for k,v in ordered_slices:
        # get data from contour dict
        if i in contour_dict:
            images.append(contour_dict[i][0])
            contours.append(contour_dict[i][1])
        # get data from dicom.read_file
        else:
            img_arr = dicom.read_file(all_slice_path[i][0]).pixel_array
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append(contour_arr)

    return np.array(images), np.array(contours)

def get_slice_data_only(path):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
    """
    all_slice_path, all_z = get_all_z(path)
    images = []

    # handle `\\` missing
    if path[-1] != '\\': path += '\\'

    # get slice orders
    ordered_slices = slice_order(path)
    
    for i in range(len(ordered_slices)):
    
#    for k,v in ordered_slices get data from dicom.read_file
        img_arr = dicom.read_file(all_slice_path[i][0]).pixel_array
        images.append(img_arr)

    return np.array(images)

def get_contour_dict(contour_file, path):
    """
    Returns a dictionary as k: img fname, v: [corresponding img_arr, corresponding contour_arr]
    Inputs:
        contour_file: .dcm contour file name with full path

    Returns:
        contour_dict: dictionary with 2d np.arrays
    """
    # handle `\\` missing
    # img_arr, contour_arr, img_fname
    contour_list = cfile2pixels(contour_file, path)

    contour_dict = {}
    for img_arr, contour_arr, z_loc in contour_list:
        contour_dict[z_loc] = [img_arr, contour_arr]

    return contour_dict

def coord2pixels(contour_dataset, sorted_slices,all_z):
    """
    Given a contour dataset (a DICOM class) and path that has .dcm files of
    corresponding images. This function will return img_arr and contour_arr (2d image and contour pixels)
    Inputs
        contour_dataset: DICOM dataset class that is identified as (3006, 0016)  Contour Image Sequence
        path: string that tells the path of all DICOM images
    Return
        img_arr: 2d np.array of image with pixel intensities
        contour_arr: 2d np.array of contour with 0 and 1 labels
    """
    contour_coord = contour_dataset.ContourData
    # x, y, z coordinates of the contour in mm
    coord = []
    for i in range(0, len(contour_coord), 3):
        coord.append((contour_coord[i], contour_coord[i + 1], contour_coord[i + 2]))
    
    z_loc = all_z.index(coord[0][2])
    # extract the image id corresponding to given countour
    # read that dicom file
#    return contour_dataset

    file_name = sorted_slices[z_loc][0]
    img = dicom.read_file(file_name)
    img_arr = img.pixel_array

    # physical distance between the center of each pixel
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])

    # this is the center of the upper left voxel
    origin_x, origin_y, _ = img.ImagePositionPatient

    # y, x is how it's mapped
    pixel_coords = [(np.ceil((y - origin_y) / y_spacing), np.ceil((x - origin_x) / x_spacing)) for x, y, _ in coord]

    # get contour data for the image
    rows = []
    cols = []
    for i, j in list(set(pixel_coords)):
        rows.append(i)
        cols.append(j)
    contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8, shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    return img_arr, contour_arr, z_loc


def cfile2pixels(file, path):
    """
    Given a contour file and path of related images return pixel arrays for contours
    and their corresponding images.
    Inputs
        file: filename of contour
        path: path that has contour and image files
    Return
        contour_iamge_arrays: A list which have pairs of img_arr and contour_arr for a given contour file
    """
    # handle `\\` missing
    if path[-1] != '\\': path += '\\'
    
    sorted_slices,all_z = get_all_z(path)
    
    f = dicom.read_file(file)
    ROIContourSeq = get_roi_names(f).index('GTV-1')

    GTV = f.ROIContourSequence[ROIContourSeq]
    # get contour datasets in a list
    contours = [contour for contour in GTV.ContourSequence]
    img_contour_arrays = [coord2pixels(cdata,sorted_slices,all_z) for cdata in contours]  # list of img_arr, contour_arr, im_id

    # debug: there are multiple contours for the same image indepently
    # sum contour arrays and generate new img_contour_arrays
    contour_dict = defaultdict(int)
    for im_arr, cntr_arr, z_loc in img_contour_arrays:
        contour_dict[z_loc] += cntr_arr
    image_dict = {}
    for im_arr, cntr_arr, z_loc in img_contour_arrays:
        image_dict[z_loc] = im_arr
    img_contour_arrays = [(image_dict[k], contour_dict[k], k) for k in image_dict]

    return img_contour_arrays

def fill_contour(contour_arr):
    # get initial pixel positions
    pixel_positions = np.array([(i, j) for i, j in zip(np.where(contour_arr)[0], np.where(contour_arr)[1])])

    # LEFT TO RIGHT SCAN
    row_pixels = defaultdict(list)
    for i, j in pixel_positions:
        row_pixels[i].append((i, j))

    for i in row_pixels:
        pixels = row_pixels[i]
        j_pos = [j for i, j in pixels]
        for j in range(min(j_pos), max(j_pos)):
            row_pixels[i].append((i, j))
    pixels = []
    for k in row_pixels:
        pix = row_pixels[k]
        pixels.append(pix)
    pixels = list(set([val for sublist in pixels for val in sublist]))

    rows, cols = zip(*pixels)
    contour_arr[rows, cols] = 1

    # TOP TO BOTTOM SCAN
    pixel_positions = pixels  # new positions added
    row_pixels = defaultdict(list)
    for i, j in pixel_positions:
        row_pixels[j].append((i, j))

    for j in row_pixels:
        pixels = row_pixels[j]
        i_pos = [i for i, j in pixels]
        for i in range(min(i_pos), max(i_pos)):
            row_pixels[j].append((i, j))
    pixels = []
    for k in row_pixels:
        pix = row_pixels[k]
        pixels.append(pix)
    pixels = list(set([val for sublist in pixels for val in sublist]))
    rows, cols = zip(*pixels)
    contour_arr[rows, cols] = 1
    return contour_arr

def create_image_mask_files(path, write_path, img_format='.png'):
    """
    Create image and corresponding mask files under to folders '\\images' and '\\masks'
    in the parent directory of path.
    
    Inputs:
        path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
        img_format (str): image format to save by, png by default
    """
    # Extract Arrays from DICOM
    X, Y = get_data(path)
    Y = np.array([fill_contour(y) if y.max() == 1 else y for y in Y])

    # Create images and masks folders
    os.makedirs(write_path + '\\images\\', exist_ok=True)
    os.makedirs(write_path + '\\masks\\', exist_ok=True)
    for i in range(len(X)):
        plt.imsave(write_path + '\\images\\' + str(i) + img_format, X[i, :, :], cmap = 'gray')
        plt.imsave(write_path + '\\masks\\' + str(i) + img_format, Y[i, :, :], cmap = 'gray')
        
def create_only_image_files(path, write_path, img_format='.png'):
    """
    Create image files under to folders '\\images' in the parent directory of path.
    
    Inputs:
        path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
        img_format (str): image format to save by, png by default
    """
    # Extract Arrays from DICOM
    X = get_slice_data_only(path)

    # Create images  folders
    os.makedirs(write_path + '\\images\\', exist_ok=True)
    for i in range(len(X)):
        plt.imsave(write_path + '\\images\\' + str(i) + img_format, X[i, :, :], cmap = 'gray')

