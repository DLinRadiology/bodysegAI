save_img = True #this will save the middle image of the stack. This is to make sure the rotation is correct.

import numpy as np
import os
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
#import dicom2nifti
import nibabel as nib
import SimpleITK as sitk
import shutil
from skimage.transform import resize
if save_img:
    import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

weightsname = 'weights'

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
act='tanh'
def get_unet(lr = 1e-5):
    inputs = Input((512, 512, 1))

    conv1 = Conv2D(16, (3, 3), activation=act, padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation=act, padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation=act, padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation=act, padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation=act, padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation=act, padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation=act, padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation=act, padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv45 = Conv2D(256, (3, 3), activation=act, padding='same')(pool4)
    conv45 = Conv2D(256, (3, 3), activation=act, padding='same')(conv45)
    pool45 = MaxPooling2D(pool_size=(2, 2))(conv45)

    conv55 = Conv2D(512, (3, 3), activation=act, padding='same')(pool45)
    conv55 = Conv2D(512, (3, 3), activation=act, padding='same')(conv55)
    pool55 = MaxPooling2D(pool_size=(2, 2))(conv55)

    conv56 = Conv2D(1024, (3, 3), activation=act, padding='same')(pool55)
    conv56 = Conv2D(1024, (3, 3), activation=act, padding='same')(conv56)
    
    up54 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv56), conv55], axis=3)
    conv57 = Conv2D(512, (3, 3), activation=act, padding='same')(up54)
    conv57 = Conv2D(512, (3, 3), activation=act, padding='same')(conv57)
    
    up55 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv57), conv45], axis=3)
    conv58 = Conv2D(256, (3, 3), activation=act, padding='same')(up55)
    conv58 = Conv2D(256, (3, 3), activation=act, padding='same')(conv58)

    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv58), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation=act, padding='same')(up6)
    conv6 = Conv2D(128, (3, 3), activation=act, padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation=act, padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation=act, padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation=act, padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation=act, padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation=act, padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation=act, padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=Adam(lr=lr), loss=dice_coef_loss, metrics=[dice_coef])
    return model
model = get_unet()
model.load_weights(os.path.join(ROOT_DIR,'xRobotstuffx/',weightsname))

def process_prediction(pred, orig_imagedata):
    # takes prediction (512x512x3) and original image with "real" HU values
    # returns originally shaped processed precition (n x n with unique values 0,1,2,5,7)
    
    #if needed, the "pred" becomes resized to original shape
    
    orig_shape = (orig_imagedata.shape)
    
    if orig_shape!=(512,512):
        print("reshaping prediction to", orig_imagedata.shape)
        
        resized_pred = np.zeros((orig_imagedata.shape[0], orig_imagedata.shape[1], 3))
        for i in range(3):
            resized_pred[:,:,i] = resize(pred[:,:,i], orig_shape, anti_aliasing="False", 
                                         preserve_range=True, mode='constant')
        pred = resized_pred
    
    #do thresholding based on original slice
    pred[:,:,0][orig_imagedata<-190]=0
    pred[:,:,0][orig_imagedata>150]=0

    pred[:,:,1][orig_imagedata<-150]=0
    pred[:,:,1][orig_imagedata>-50]=0

    pred[:,:,2][orig_imagedata<-190]=0
    pred[:,:,2][orig_imagedata>-30]=0

    #get a max for prediction all axes
    maximum = np.max(pred, axis=2)

    #to not be true on 0, make everything not relevant to lower than zero
    maximum[maximum<0.5]=-1

    orig_size_pred1ch = np.zeros(orig_shape)
    orig_size_pred1ch[pred[:,:,0]==maximum] = 1
    orig_size_pred1ch[(orig_size_pred1ch==1) & (orig_imagedata<=-30)] = 2
    orig_size_pred1ch[pred[:,:,1]==maximum] = 5
    orig_size_pred1ch[pred[:,:,2]==maximum] = 7
    
    return orig_size_pred1ch

for filename in os.listdir(os.path.join(ROOT_DIR,'input/')):

    if 'nii' in filename:
        
        print("working on file", filename)
        
        #loading file
        volumedata = nib.load(os.path.join(ROOT_DIR,'input/',filename))
        affine = volumedata.affine
        volumedata = volumedata.get_fdata()
        
        #saving copy for comparison to raw
        orig_volumedata = np.copy(volumedata)
        
        #exception for if the nii file is 2D
        if len(orig_volumedata.shape)==2:
            volumedata = volumedata[:,:,np.newaxis]
        
        #thresholding, normalizing, and rotating
        volumedata[volumedata < -400] = -400
        volumedata[volumedata > 600] = 600
        volumedata = volumedata + 535.7372827952495
        volumedata = volumedata / 492.83128067388367
        volumedata = np.rot90(volumedata, axes=(0,1), k=3)
        for i in range(volumedata.shape[2]):
            volumedata[:,:,i] = np.fliplr(volumedata[:,:,i])

        #to check if correct rotation before predicting
        if save_img:
            plt.figure(figsize=(15,15))
            middle_slice = int(volumedata.shape[2]/2)
            plt.imshow(volumedata[:,:,middle_slice], cmap='gray')
            plt.savefig(os.path.join(ROOT_DIR,"xRobotstuffx/nii_img.png"))
        
        #creating empty prediction volume after resizing
        predictionvolume = np.zeros(orig_volumedata.shape)
        if len(orig_volumedata.shape)==2:
            predictionvolume = predictionvolume[:,:,np.newaxis]

        #predicting each slice
        for eachslice in range(volumedata.shape[2]):
                            
            #get original slice HU values for thresholding. Rotate it like the data being processed.
            #the following check is for 2d vs. 3d nifti data
            if len(orig_volumedata.shape)==2:
                orig_imagedata = orig_volumedata
            else:
                orig_imagedata = orig_volumedata[:,:,eachslice]
            orig_imagedata = np.rot90(orig_imagedata, k=3)
            orig_imagedata = np.fliplr(orig_imagedata)

            #resizes if needed
            orig_shape = (orig_imagedata.shape)
            if orig_shape != (512,512):
                image_to_predict = resize(volumedata[:,:,eachslice], (512,512), anti_aliasing="False", 
                                         preserve_range=True, mode='constant')
            else:
                image_to_predict = volumedata[:,:,eachslice]
            
            #creates prediction
            pred = model.predict(image_to_predict[np.newaxis, :, :, np.newaxis])
            pred = pred[0]

            
            
            orig_size_pred1ch = process_prediction(pred, orig_imagedata)
    
            to_add = np.fliplr(orig_size_pred1ch)
            predictionvolume[:,:,eachslice] = np.rot90(to_add, k=1)
            
            print("segmented slice", eachslice+1, "of", predictionvolume.shape[2])
            
                    
        #exception for if the nii file is 2D
        if len(orig_volumedata.shape)==2:
            predictionvolume = predictionvolume[:,:,0]
        
        #save the file
        savefile = nib.Nifti1Image(predictionvolume, affine)
        nib.save(savefile, os.path.join(ROOT_DIR,'output/s'+filename))

    if os.path.isdir(os.path.join(ROOT_DIR,'input/')+filename) and filename.startswith('tag'):

        #folder with tags must start with "tag"

        for filename2 in os.listdir(os.path.join(ROOT_DIR,'input/')+filename):
            if filename2.endswith('tag'):

                #copy the dicom file
                shutil.copy(os.path.join(ROOT_DIR,'input/')+filename+'/'+filename2[:-4], os.path.join(ROOT_DIR,'output/')+filename2[:-4])

                #read and process the dicom file
                image = sitk.ReadImage(os.path.join(ROOT_DIR,'input/')+filename+'/'+filename2[:-4])
                imagearray = sitk.GetArrayFromImage(image)[0,:,:]
                orig_imagedata = np.copy(imagearray)
                print('bildedimensjoner: '+str(imagearray.shape))
                
                #resizing wrong shape
                if (imagearray.shape != (512,512)):
                    originalshape = imagearray.shape
                    imagearray = resize(imagearray, (512,512), preserve_range=True, anti_aliasing="False", mode='constant')
                
                #normalizing
                imagearray[imagearray < -400] = -400
                imagearray[imagearray > 600] = 600
                imagearray = imagearray + 535.7372827952495
                imagearray = imagearray / 492.83128067388367

                sliceprediction = model.predict(imagearray[np.newaxis, :, :, np.newaxis])[0]

                
                sliceprediction = process_prediction(sliceprediction, orig_imagedata)
                
                #import and save tag
                tagfile = np.fromfile(os.path.join(ROOT_DIR,'input/')+filename+'/'+filename2, dtype='int8', sep="")
                tagfile = tagfile[:288]
                A = np.concatenate([tagfile, sliceprediction.flatten()])
                A = A.astype('int8')
                A.tofile(os.path.join(ROOT_DIR,'output/')+filename2, sep="" ) #should make .tag files work too

print('Ferdig')
