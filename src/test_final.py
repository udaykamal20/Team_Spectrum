# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 03:00:35 2018

@author: DSP Research Lab
"""

    
""" Please provide the following paths before running the code

*model_weightpath = path of the weightfile for the segmentation model    
*test_data_folder_path = path of the folder containing all the test/validation patient folders
*csvfile_savepath = path where generated csv file will be created (for validation data)
*test_mask_save_path = path where binary masks for the test patients will be created (for test data)
*temporary_path = path of an EMPTY folder which will contain the newly created images(for both training and validation)  and masks(for validation only) for further processing.

Please note that, during testing, our model will generate some temporary folders where binary masks from ground truth(if test data is validation set)
and the slice images will be saved in the corresponding patient directory. After generating the result, this script will automatically
remove those newly created folders completely.
 
"""


from model_train_final import get_3D_Recurrent_DenseUnet
from generate_data_from_dicom import create_only_image_files,create_image_mask_files,decide_validation_or_test
from tqdm import tqdm
import re,glob,os,cv2
import numpy as np
import matplotlib.pyplot as plt
from get_metrics import mean_surface_distance,hausdorff_distance_95
import pandas as pd

img_rows = 256
img_cols = 256
depth = 8
batch_size = 1

    
    
model_weightpath = 'Recurrent_3D_DenseUnet_model.h5'

test_data_folder_path = 'G:\\VIP_CUP18_ValidationData\\VIP_CUP18_ValidationData\\' #Folder path where all 
                                                        #the val/test patients are located
                                                        
csvfile_savepath = 'G:\\Saved_Result\\'                 #Folder path where the generated csv 
                                                        #file will be saved
                                                        
test_mask_save_path = 'G:\\Saved_Result\\'              #Folder path where the generated binary 
                                                        #masks for each test patient will be 
                                                        #saved
                                                        
temporary_path = 'G:\\Temporary_Folder\\'                   #Path of the Directory where the 
                                                        #temporarily generated images will be 
                                                        #saved


def create_dataset(valid_or_test):

    all_patientdir = os.listdir(test_data_folder_path)
    for i in all_patientdir:           
        os.makedirs(temporary_path + i, exist_ok=True)
        if valid_or_test:
            #its a validation datapath, so create both image and masks
            create_image_mask_files(test_data_folder_path + i, temporary_path + i)
        else:
            #its a test datapath, so create both image only
            create_only_image_files(test_data_folder_path + i, temporary_path + i)    



def sorted_nicely(l):    
    #sort a list numerically 
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def get_img_list_from_patient_folder(path,patient, depth = 8):
    #generate list of depth image patches 
    depth_img = []
    all_img = sorted_nicely(glob.glob(path + patient +  '\\images\\*'))
    for k in range(len(all_img)-depth+1):  
        depth_img+=[(all_img[k:k+depth])]
    return depth_img

def get_all_mask(patient,path):
    all_mask_path = sorted_nicely(glob.glob(path + patient +  '\\masks\\*'))
    all_mask = np.zeros((512,512,len(all_mask_path)),dtype = np.int32)
    
    for k in range(len(all_mask_path)):  
        all_mask[:,:,k]= cv2.imread(all_mask_path[k], cv2.IMREAD_GRAYSCALE)
        all_mask[:,:,k] = np.asarray(all_mask[:,:,k]/255, dtype = np.int32)
    return all_mask


def get_batch_image_from_list(img_list,idx,depth=depth):
    ### Get batch image arrays from list of paths
    batch_images = np.zeros((batch_size,img_rows, img_cols,depth,1), dtype = np.float64)
    all_img = np.zeros((img_rows, img_cols, depth))
    list_of_slices = img_list[idx]
    for i in range(depth):        
        img_name = list_of_slices[i]
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_rows, img_cols))
        img = np.asarray(img, dtype = np.float32)
        all_img[:,:,i] = img        
    all_img = all_img/np.max(all_img)
    batch_images[0] = np.expand_dims(all_img, axis = -1)
    return batch_images

def save_pred(all_pred,datapath,patientdir):

    if os.path.exists(datapath + patientdir)==False:
        os.mkdir(datapath + patientdir)
        
    for i in range(all_pred.shape[-1]):
        plt.imsave(datapath + patientdir + "\\" + str(i)+'.png', all_pred[:,:,i], cmap = 'gray')
        
def select_threshold(pred_from_3d, initial_th=0.4):
    
    all_pred_flat = pred_from_3d.flatten()
    p = all_pred_flat[all_pred_flat>initial_th]
    histogram = np.histogram(p,bins = 1000)
    AAA = np.copy(histogram[0])
    AAA.sort()
    A = histogram[1][np.where(histogram[0] == AAA[-3])]
    B = histogram[1][np.where(histogram[0] == AAA[-4])]
    C = histogram[1][np.where(histogram[0] == AAA[-5])]
    D = histogram[1][np.where(histogram[0] == AAA[-6])]
    out = 0.4*np.average(A) + 0.3*np.average(B) + 0.2*np.average(C)+ 0.1*np.average(D)
    return np.average(out)


def get_dice_with_fp_fn_hd_95(y_true, y_pred):
    
    ## here for all false positive,false negative and slices with no tumors
    ## we ignore the c2,c3 scores by assigning them a value of -1, and c1 with 1 for slice with no tumor and 0 for slice with false positive and false negative which will be further removed
    ## we calculate all the metrics for the cases where our model's prediction has an overlap with
    ## the ground truth tumors because otherwise the C2 and C3 scores will have infinite result which will make the overall result infinite.
    
    ## for convinience, we also calculate false positive and false negative case as well
    
    fp = 0
    fn = 0
    
    ##add dilation on the predicted mask
    y_pred = np.asarray(y_pred,np.uint8)
    y_pred = cv2.dilate(y_pred,kernel,iterations = 1)
    y_pred = np.int32(y_pred>0.001)

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    y_true_s = np.sum(y_true_f)
    y_pred_s = np.sum(y_pred_f)
    
    if y_true_s==0 and y_pred_s==0:
        dice = 1
        c2 = -1
        c3 = -1
        return [dice,c2,c3,fp,fn]
    elif y_true_s==0 and y_pred_s!=0:
        fp+=1
        dice = 0
        c2 = -1
        c3 = -1
        return [dice,c2,c3,fp,fn]
    elif y_true_s!=0 and y_pred_s==0:
        fn+=1
        dice = 0
        c2 = -1
        c3 = -1
        return [dice,c2,c3,fp,fn]
    
    intersection = np.sum(y_true_f * y_pred_f)  
    if intersection==0 and fn==0:
        fn+=1
        dice = 0
        c2 = -1
        c3 = -1
    elif intersection!=0:      
        dice = (2. * intersection) / (y_true_s + y_pred_s)
        c2 = mean_surface_distance(y_true, y_pred)
        c3 = hausdorff_distance_95(y_true, y_pred)
    
    return [dice,c2,c3,fp,fn]


def get_all_pred(img_list,depth,num=10):
    # generate prediction from a list of depth image path for a patient
    # we intentionally ignore the first and last 10 slices as they cover far away area of our body from the lung which are very unlikely to contain any tumor. 
    # So we ignore them to reduce the processig time.
    all_pred = np.zeros((img_rows,img_rows,len(img_list)+depth-1))

    for i in range(len(img_list)):
        #we average the probabilities for the overlapping volume patches
        if i>num-1 and i<len(img_list)+depth-num:
            img = get_batch_image_from_list(img_list,i)
            if i!=num:
                all_pred[:,:,i:i+depth] = all_pred[:,:,i:i+depth] + model.predict(img)[0,:,:,:,0]
            else:
                all_pred[:,:,i:i+depth] = model.predict(img)[0,:,:,:,0] 
        else:
            continue
    
    for j in range(all_pred.shape[-1]):
        if j<=num or j>=len(img_list)+depth-num:
            continue
        elif j>num and j<(num+depth):
            all_pred[:,:,j] = all_pred[:,:,j]/(j-num)
        elif j>=(num+depth) and j<len(img_list)-num:
            all_pred[:,:,j] = all_pred[:,:,j]/depth
            
        elif j>=len(img_list)-num:
            all_pred[:,:,j] = all_pred[:,:,j]/(all_pred.shape[-1]-num-j+1)    
    
    return all_pred    


#testing starts here

if __name__ == "__main__":
    
    #define kernel for dilation, disk kernel with radius 3
    a, b = 1, 1
    n = 7
    r = 3
    
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    
    array = np.ones((n, n))
    array[mask] = 1
    kernel = np.asarray(array, dtype = np.uint8)
    

    if test_data_folder_path[-1] != '\\': test_data_folder_path += '\\'  
    if csvfile_savepath[-1] != '\\': csvfile_savepath += '\\' 
    if test_mask_save_path[-1] != '\\': test_mask_save_path += '\\'
    
    print('Creating model')
    model = get_3D_Recurrent_DenseUnet()
    model.load_weights(model_weightpath)

    print('Deciding whether its validation or test data')
    valid_or_test = decide_validation_or_test(test_data_folder_path)
    
    if valid_or_test:
        all_scores = []
        print('Generating CSV file on validation data...')
    else:
        print('Generating binary mask on test data...')
        
    print("Now creating data from DICOM files..." )    
#    create_dataset(valid_or_test)
  
    
    print("Data generation complete" )
    patientdir = os.listdir(temporary_path)
    
    print("Now generating predictions....")
    #generate precition probabilities
    for i in tqdm(range(len(patientdir))):
        #get list of depth images from patient folder
        lis = get_img_list_from_patient_folder(temporary_path,patientdir[i])
        num = int(0.2*(len(lis)+depth))
        all_pred = get_all_pred(lis, depth,num)
        all_pred = cv2.resize(all_pred, (512, 512))

        th=0.7

        all_pred = np.int32(all_pred>th)
        
        if valid_or_test==0:
            save_pred(all_pred,test_mask_save_path,patientdir[i])
        else:
            label_mask = get_all_mask(patientdir[i],temporary_path)
            indiv_patient_score = [get_dice_with_fp_fn_hd_95(label_mask[:,:,idx],all_pred[:,:,idx]) for idx in range(all_pred.shape[-1])]
            all_scores.append(indiv_patient_score)
    
    
    if valid_or_test==1:
        print("Prediction complete, now calculating the metrics...")
        iou_all = []
        c2_all = []
        c3_all = []
        fp_all = []
        fn_all = []
        for x,scores  in  enumerate(all_scores):

            p = np.array(scores)
            iou_per_slice = p[:,0]
            c2_per_slice = p[:,1]
            c3_per_slice = p[:,2]
            
            iou_avg = np.average(p[:,0][np.logical_and(p[:,0]!=1, p[:,0]!=0)])
            c2_avg = np.average(p[:,1][p[:,1]!=-1])
            c3_avg = np.average(p[:,2][p[:,2]!=-1])
            fp_per_patient = np.sum(p[:,3])
            fn_per_patient = np.sum(p[:,4])
            
            iou_all.append(iou_avg)
            c2_all.append(c2_avg)
            c3_all.append(c3_avg)
            fp_all.append(fp_per_patient)
            fn_all.append(fn_per_patient)


            slice_no = list(map(str,np.arange(0,len(p))))
            
            slice_list =['Slice_' + i for i in slice_no]
            
            Metric_per_patient = pd.DataFrame(
                    {'Slice_number':    slice_list,
                     'All C1':          iou_per_slice,
                     'All C2':          c2_per_slice,
                     'All C3':          c3_per_slice})
    
            Metric_per_patient.to_csv(csvfile_savepath + patientdir[x] + '_result.csv',index=False,columns = ['Slice_number','All C1','All C2','All C3'])
        
        ## We hightlight the completely missed patient cases as 'No Detection'
        Metric_list = pd.DataFrame(
        {'PatientID':   patientdir,
         'Avg C1':          iou_all,
         'Avg C2':          c2_all,
         'Avg C3':          c3_all,
         'Misdetection':    fp_all,
         'Missed detection':fn_all})

      
        Metric_list.to_csv(csvfile_savepath + 'Spectrum.csv',index=False,columns = ['PatientID','Avg C1','Avg C2','Avg C3','Misdetection','Missed detection'])
        
        print("CSV file created!")
        
    else:
        print("Binary masks have been saved")
    
    #if you want to delete all the temporary files which have been created during testing you can uncomment and run the following codes:
    #delete all the temporarily created images and masks    
    
    ##os.remove(temporary_path)