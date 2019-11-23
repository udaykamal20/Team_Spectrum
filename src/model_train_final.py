
#%%

#necessary libraries

from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, Conv2DTranspose, SpatialDropout3D, ConvLSTM2D, TimeDistributed
from keras.layers.core import Activation, Permute
from keras.layers.convolutional import MaxPooling2D  
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import random
import cv2
import pandas as pd
import glob,os
import re
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage
import keras
from sklearn.model_selection import KFold

from imgaug_master.imgaug import augmenters as iaa #augmenters function is saved in the imgaug_master folder's
                                                    #imgaug.py file

#%%

"""Both trainpath and valpath must have the following directory tree: (here for)

train:. 
│
├───images
│   │   LUNG1-010
│   │   LUNG1-011
│   │   LUNG1-012
│   │   .......
│   
├───masks
│   │   LUNG1-010
│   │   LUNG1-011
│   │   LUNG1-012
│   │   .......
│   │              


val:. 
│
├───images
│   │   LUNG1-000
│   │   LUNG1-001
│   │   LUNG1-002
│   │   .......
│   
├───masks
│   │   LUNG1-000
│   │   LUNG1-001
│   │   LUNG1-002
│   │   .......
│   │              


"""

#%%

img_rows = 256
img_cols = 256
depth = 8
smooth = 1.
nb_epoch = 30
batch_size = 2
n_fold = 5
Epochs = 30

#%%
#function for augmentation
def create_augmentor():
    intensity_seq = iaa.Sequential([
        iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
        iaa.OneOf(
            [iaa.Sequential([
                iaa.OneOf([
                    iaa.Add((-10, 10)),
                    iaa.AddElementwise((-10, 10)),
                    iaa.Multiply((0.9, 1.1)),
                    iaa.MultiplyElementwise((0.95, 1.05)),
                ]),
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.0, 1.0)),
                iaa.AverageBlur(k=(2, 4)),
                iaa.MedianBlur(k=(3, 5))
            ])
        ])
    ], random_order=False)
    return intensity_seq



def random_elastic_deformation(image, alpha, sigma, mode='nearest',
                               random_state=None):

    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width, channels = image.shape

    dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    indices = (np.repeat(np.ravel(x+dx), channels),
               np.repeat(np.ravel(y+dy), channels),
               np.tile(np.arange(channels), height*width))

    values = map_coordinates(image, indices, order=1, mode=mode)

    return values.reshape((height, width, channels))


def rotate_3D(image,mask, angle, axes=(0,1)):
    rotated_image = scipy.ndimage.interpolation.rotate(image, angle,axes, mode = 'nearest',reshape=False)
    rotated_mask = scipy.ndimage.interpolation.rotate(mask, angle, axes,mode = 'nearest', reshape=False)
    return rotated_image, rotated_mask

#%%

#functions for creating the train & val data
 
def sorted_nicely( l ):
    
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)  


#depth data list create
def create_list_of_depth_data(trainpath,valpath,depth, seed = 200):
    
    train_list = os.listdir(trainpath)
    val_list = os.listdir(valpath)

    depth_train_img = []
    depth_train_mask = []    
    
    depth_val_img = []
    depth_val_mask = [] 


    print('training data create started')
    
    for i in range(len(train_list)):
        p = sorted_nicely(glob.glob(trainpath + train_list[i] + '\\masks\\*'))
        train_img = []
        train_mask = []
        demo = 0
        for j in range(len(p)):
            q = cv2.imread(p[j], cv2.IMREAD_GRAYSCALE)
            #we only take the slice with tumors
            if np.min(q)!=np.max(q):
                if demo and (j-demo)!=1:
                    for z in range(j-demo-1):
                        train_mask.append(p[demo+z+1])
                        train_img.append('\\'.join(p[demo+z+1].split('\\')[:-2])+'\\images\\'+p[j].split('\\')[-1].split('_')[-1])
                        
                train_mask.append(p[j])
                train_img.append('\\'.join(p[j].split('\\')[:-2])+'\\images\\'+p[j].split('\\')[-1].split('_')[-1])
                demo = np.copy(j)
         
        #if any tumor consists of less than depth variable, then we take additional subsequent non-tumor slices to make the complete volume data    
        if len(train_img)<depth:
            for j in range(depth-len(train_img)):
                train_mask.append(p[demo + j + 1])
                train_img.append('\\'.join(p[demo+1+j].split('\\')[:-2])+'\\images\\'+p[demo+1+j].split('\\')[-1].split('_')[-1])
        
        # now we create the datapath list of the overlapping depth images        
        for k in range(len(train_img)):
            
            if k and k+depth>len(train_img):
                break 
            
            elif k+depth>len(train_img):
                depth_mask = train_mask[k:k+depth]
                depth_img = train_img[k:k+depth]
                depth_train_img.append(depth_img)
                depth_train_mask.append(depth_mask)
                break
                
            depth_mask = train_mask[k:k+depth]
            depth_img = train_img[k:k+depth]
            depth_train_img.append(depth_img)
            depth_train_mask.append(depth_mask)
        
    print('train data created')
    
    # now we use the same approach for generating validation data
    for i in range(len(val_list)):
        p = sorted_nicely(glob.glob(valpath + val_list[i] + '\\masks\\*'))
        val_img = []
        val_mask = []
        demo=0
        for j in range(len(p)):
            q = cv2.imread(p[j], cv2.IMREAD_GRAYSCALE)
            if np.min(q)!=np.max(q):
                if demo and (j-demo)!=1:
                    for z in range(j-demo-1):
                        val_mask.append(p[demo+z+1])
                        val_img.append('\\'.join(p[demo+z+1].split('\\')[:-2])+'\\images\\'+p[j].split('\\')[-1].split('_')[-1])

                val_mask.append(p[j])
                val_img.append('\\'.join(p[j].split('\\')[:-2])+'\\images\\'+p[j].split('\\')[-1].split('_')[-1])
                demo = np.copy(j)
        if len(val_img)<8:
            for j in range(8-len(val_img)):
                val_mask.append(p[demo + j + 1])
                val_img.append('\\'.join(p[demo+1+j].split('\\')[:-2])+'\\images\\'+p[demo+1+j].split('\\')[-1].split('_')[-1])
                                
        for k in range(len(val_img)):
            
            if k and k+depth>len(val_img):
                break 
            
            elif k+depth>len(val_img):
                depth_mask = val_mask[k:k+depth]
                depth_img = val_img[k:k+depth]
                depth_val_img.append(depth_img)
                depth_val_mask.append(depth_mask)
                break       
            
            depth_mask = val_mask[k:k+depth]
            depth_img = val_img[k:k+depth]
            depth_val_img.append(depth_img)
            depth_val_mask.append(depth_mask)
                
    print('val data created')
    
    #randomly shuffle the train and validation data
    combined_train = list(zip(depth_train_img, depth_train_mask))
    random.shuffle(combined_train)  
    depth_train_img[:], depth_train_mask[:] = zip(*combined_train)
    
    combined_val = list(zip(depth_val_img, depth_val_mask))
    random.shuffle(combined_val)  
    depth_val_img[:], depth_val_mask[:] = zip(*combined_val)
    
    
    return depth_train_img, depth_train_mask, depth_val_img, depth_val_mask 
#%%
#preparing each volume of slice to feed to the batch generator
def get_image(img_list,mask_list,depth):

    all_img = np.zeros((img_rows, img_cols, depth), dtype = np.uint8)
    all_mask = np.zeros((img_rows, img_cols, depth), dtype = np.uint8)
    
    for i in range(depth):      
        
        img_name = img_list[i]
        mask_name = mask_list[i]
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
               
        #resize to 256x256 size 
        img = cv2.resize(img, (img_rows, img_cols))
               
        #generate binary mask
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(mask, np.min(mask), 1, cv2.THRESH_BINARY)[1]        
        
        mask = cv2.resize(mask, (img_rows, img_cols))
        
        all_img[:,:,i] = img
        all_mask[:,:,i] = mask
    
    return all_img,all_mask

#%%
#threadsafe_generator 
import threading
class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g

#takes a dataframe as input and generates batches of training data ready to be fed to the model
@threadsafe_generator 
def generate_batch_augment(d, batch_size,intensity_seq, train = True):

    batch_images = np.zeros((batch_size,img_rows, img_cols,depth,1), dtype = np.float32)
    batch_masks = np.zeros((batch_size, img_rows, img_cols,depth,1), dtype = np.float32)
    
    while 1:
        #shuffle the data during after each epoch
        d.sample(frac=1).reset_index(drop=True)
        for i_line in range(0, len(d), batch_size): 
            for i_batch in range(batch_size):
                if (i_line+i_batch)>=len(d):
                    continue

                depth_img_path = d['imgpath'][d.index[i_line+i_batch]]                
                depth_mask_path = d['maskpath'][d.index[i_line+i_batch]]
                
                img,mask = get_image(depth_img_path,depth_mask_path, depth)

                if train:
                    #augment the batches in a random order
                    param = np.random.randint(0,5)
                    if param==0:
                        img = img
                        mask = mask
                    elif param==1:
                        angle = np.random.randint(-5,5)
                        img,mask = rotate_3D(img, mask, angle)
                    elif param ==2:
                        stacked = np.concatenate((img, mask), axis=2)
                        augmented = random_elastic_deformation(stacked, stacked.shape[1] * 2, stacked.shape[1] * 0.08)
                        img = augmented[:,:,:depth]
                        mask = augmented[:,:,depth:]
                    elif param ==3:
                        img = intensity_seq.augment_image(img)
                        mask = mask
                    else:
                        stacked = np.concatenate((img, mask), axis=2)
                        augmented = np.fliplr(stacked)
                        img = augmented[:,:,:depth]
                        mask = augmented[:,:,depth:]
                
                img = np.asarray(img,dtype = np.float32)
                img = img/np.max(img)
                batch_images[i_batch] = np.expand_dims(img, axis = -1)
                batch_masks[i_batch] = np.expand_dims(mask, axis = -1)
                
            if(i_line+i_batch)>=len(d):
                continue

            yield batch_images, batch_masks


#%%
#loss function

def loss_function(y_true, y_pred):
    alpha = 0.5 #to control fp
    beta  = 0.5 #to control fn
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl -T
#%%
    
#function for dice coefficient calculation 

def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#%%

#model definittion

def get_3D_Recurrent_DenseUnet():
    
    inputs = Input((img_rows, img_cols, depth, 1))
    
    #list of number of filters per block
    depth_cnn = [32, 64, 128, 256]
    
    ##start of encoder block
    
    ##encoder block1
    conv11 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same')(inputs)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same')(conc11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conc12 = concatenate([inputs, conv12], axis=4)
    perm = Permute((3,1,2,4))(conc12)
    pool1 = TimeDistributed(MaxPooling2D((2, 2)))(perm)
    pool1 = Permute((2,3,1,4))(pool1)

    pool1 = SpatialDropout3D(0.1)(pool1)

    #encoder block2
    conv21 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same')(pool1)
    conv21 = BatchNormalization()(conv21)
    conv21 = Activation('relu')(conv21)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same')(conc21)
    conv22 = BatchNormalization()(conv22)
    conv22 = Activation('relu')(conv22)
    conc22 = concatenate([pool1, conv22], axis=4)   
    perm = Permute((3,1,2,4))(conc22)
    pool2 = TimeDistributed(MaxPooling2D((2, 2)))(perm)
    pool2 = Permute((2,3,1,4))(pool2)

    pool2 = SpatialDropout3D(0.1)(pool2)

    #encoder block3
    conv31 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same')(pool2)
    conv31 = BatchNormalization()(conv31)
    conv31 = Activation('relu')(conv31)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same')(conc31)
    conv32 = BatchNormalization()(conv32)
    conv32 = Activation('relu')(conv32)
    conc32 = concatenate([pool2, conv32], axis=4)  
    perm = Permute((3,1,2,4))(conc32)
    pool3 = TimeDistributed(MaxPooling2D((2, 2)))(perm)

    pool3 = SpatialDropout3D(0.1)(pool3)
    
    ##end of encoder block
    
    #ConvLSTM block 
    x = BatchNormalization()(ConvLSTM2D(filters =depth_cnn[3], kernel_size = (3,3), padding='same', return_sequences=True)(pool3))
    x = BatchNormalization()(ConvLSTM2D(filters =depth_cnn[3], kernel_size = (3,3), padding='same', return_sequences=True)(x))
    x = BatchNormalization()(ConvLSTM2D(filters = depth_cnn[3], kernel_size = (3,3), padding='same', return_sequences=True)(x))


    # start of decoder block
    
    # decoder block1
    up1 = TimeDistributed(Conv2DTranspose(depth_cnn[2], (2, 2), strides=(2, 2), padding='same'))(x)   
    up1 = Permute((2,3,1,4))(up1)
    up6 = concatenate([up1, conc32], axis=4)
    conv61 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same')(up6)
    conv61 = BatchNormalization()(conv61)
    conv61 = Activation('relu')(conv61)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(depth_cnn[2], (3, 3, 3), padding='same')(conc61)
    conv62 = BatchNormalization()(conv62)
    conv62 = Activation('relu')(conv62)
    conv62 = concatenate([up6, conv62], axis=4)

    #decoder block2
    up2 = Permute((3,1,2,4))(conv62)
    up2 = TimeDistributed(Conv2DTranspose(depth_cnn[1], (2, 2), strides=(2, 2), padding='same'))(up2)
    up2 = Permute((2,3,1,4))(up2)    
    up7 = concatenate([up2, conv22], axis=4)
    conv71 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same')(up7)
    conv71 = BatchNormalization()(conv71)
    conv71 = Activation('relu')(conv71)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(depth_cnn[1], (3, 3, 3), padding='same')(conc71)
    conv72 = BatchNormalization()(conv72)
    conv72 = Activation('relu')(conv72)
    conv72 = concatenate([up7, conv72], axis=4)
    
    #decoder block3
    up3 = Permute((3,1,2,4))(conv72)
    up3 = TimeDistributed(Conv2DTranspose(depth_cnn[0], (2, 2), strides=(2, 2), padding='same'))(up3)
    up3 = Permute((2,3,1,4))(up3)
    up8 = concatenate([up3, conv12], axis=4)
    conv81 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same')(up8)
    conv81 = BatchNormalization()(conv81)
    conv81 = Activation('relu')(conv81)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(depth_cnn[0], (3, 3, 3), padding='same')(conc81)
    conv82 = BatchNormalization()(conv82)
    conv82 = Activation('relu')(conv82)
    conc82 = concatenate([up8, conv82], axis=4)

    ##end of decoder block

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc82)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss = loss_function, metrics=[dice_coef])

    return model

#%%
    
#train n_fold model with cyclic learning rate scheduler
    
def train_model(model, batch_size, epochs, kf,x, n_fold, model_name,intensity_seq):

   i = 1

   for train_index, test_index in kf.split(x):
        new_train = train_df.iloc[train_index]
        new_train = new_train.reset_index(drop=True)
        
        training_gen = generate_batch_augment(new_train,batch_size,intensity_seq,True)
        val_gen = generate_batch_augment(val_df,batch_size,intensity_seq, False)

        callbacks = [EarlyStopping(monitor='val_dice_coef', patience=5, verbose=1, min_delta=1e-6, mode = 'max'),
                     keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,patience=3, verbose=1, mode='max', cooldown=0, min_lr=1e-8),
                     ModelCheckpoint(model_name + '_' + str(i) + '.h5', monitor='val_dice_coef', verbose=1, save_weights_only = False, save_best_only=True, mode='max')]
        
        model = model
        
        model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss = loss_function, metrics=[dice_coef])
     
        model.fit_generator( 
                             generator = training_gen, 
                             steps_per_epoch  = int(len(train_img)/batch_size), 
                             epochs           = epochs, 
                             verbose          = 1,
                             validation_data  = val_gen,
                             validation_steps = int(len(val_df)/batch_size),
                             callbacks        = callbacks, 
                             workers          = 3,
                             max_queue_size   = 8)
        
        if i>1:
            model.load_weights(filepath= model_name + '_' + str(i) +'.h5')
        i += 1

        if i <= n_fold:
#            print('Now beginning training for fold {}\n\n'.format(i))
            print('Now beginning training for {} fold {}\n\n'.format(model_name,i))
        else:
            print('Finished training {}\n\n!'.format(model_name))

#%%
#training starts here
##number of folds to split the training data
### during each epoch, we take 4 folds for training 
if __name__ == "__main__":

    
    trainpath = 'G:\\VIP\\TrainPath\\'#Please enter your complete datapath of the folder containing all the training data in .png format'
    valpath = 'G:\\VIP\\ValPath\\'#'Please enter your complete datapath of the folder containing all the validation data in .png format'

    if trainpath[-1] != '\\': trainpath += '\\'  
    if valpath[-1] != '\\': valpath += '\\' 

    
    K.set_image_data_format("channels_last")
    img_rows = 256
    img_cols = 256
    depth = 8
    smooth = 1.
    batch_size = 2
    
    n_fold = 5
    Epochs = 30

    #%%
    #create the training and validation data generator 
            
    train_img,train_mask,val_img, val_mask = create_list_of_depth_data(trainpath, valpath,depth=depth)
    
    train_df = pd.DataFrame(
    {'imgpath': train_img,
     'maskpath': train_mask
    })
    
    val_df = pd.DataFrame(
    {'imgpath': val_img,
     'maskpath': val_mask
    })
        
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.sample(frac=1).reset_index(drop=True)
        
    model = get_3D_Recurrent_DenseUnet()
    print('model_created')       
    
    weightname = '3D_Recurrent_DenseUnet_model'
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    intensity_seq = create_augmentor()
    
    print('training_starrted')
    
    train_model(model, batch_size, Epochs, kfold, train_df, n_fold,weightname,intensity_seq)


