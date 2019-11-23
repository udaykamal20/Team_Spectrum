# VIP_CUP_2018_Team_Spectrum-Lung_Cancer_Radiomics-Lung_Tumor_Segmentation
This repository contains the codes and submission package of the Team Spectrum, the first runner up team of IEEE Video and Image Processing Cup 2018. The task was to come up with a lung tumor region segmentation algorithm from volume CT database.

![alt text](https://github.com/udday2014/VIP_CUP_2018_Team_Spectrum-Lung_Cancer_Radiomics-Tumor_Region_Segmentation/blob/master/VIP_CUP_everyone.jpg)


The competition data can be accessed by requesting the organizer: 
http://i-sip.encs.concordia.ca/2018VIP-Cup/index.html.

Details of the competition can be found in the following link:
https://2018.ieeeicip.org/VIPCup.asp.

The details of our approach and performance can be found in the 'Report.pdf' file. 

The following procedures are to be followed to reporduce the submitted results:

First install the required libraries by running 'pip install -r requirements.txt' on the terminal.

To run our model on the test dataset, one have to must modify the following paths in the 'test_final.py' script, whcich includes:

    * model_weightpath = '/Recurrent3D_DenseUnet_model.h5'
    * test_data_folder_path = '/VIP_CUP18_TestData/' (Folder path where all the val/test patients are located)
    * csvfile_savepath = '/Saved_Result/' (Folder path where the generated csv file will be saved)
    * test_mask_save_path = '/Saved_Result/' (Folder path where the generated binary masks for each test patient will be saved)
    * temporary_path = '/Temporary_Folder/' (Path of the Directory where the temporarily generated images will be saved)


