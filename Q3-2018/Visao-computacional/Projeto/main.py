# Reading library
import numpy as np
import cv2
import pandas as pd

import os
import sys
import time

# Keras
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K


# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.utils import class_weight

# UTILS
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
from skimage.transform import rotate

# ----------------------- PARAMETERS ---------------------------
np.random.seed(177)

# Specify GPU's to Use
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Specify parameters before model is run.
batch_size = 1000
nb_classes = 5
nb_epoch = 30

img_rows, img_cols = 256, 256
channels = 3
nb_filters = 32
kernel_size = (8,8)

dict_labels = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative DR"}

labels = "database/labels.csv"
trainLabels = "database/trainLabels.csv"
testLabels = "database/testLabels.csv"
customTrainLabels = "database/custom_trainLabels.csv"


folder_train = "database/slice-train/"
folder_test = "database/slice-test/"

folder_train_resized = "database/train-resized/"
folder_test_resized = "database/test-resized/"

file_X_npy = "database/X_train.npy"

#------------------- PRE PROCESSING FUNCTIONS --------------------

def find_black_images(file_path, df):
    """
        Creates a column of images that are not black (np.mean(img) != 0)
    """

    print("Finding Black images")
    lst_imgs = [l for l in df['image']]
    return [1 if np.mean(np.array(Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]


def rotate_images(file_path, angle, lst_imgs):
    '''
    Rotates image on angle
    INPUT
        file_path: file path to the folder containing images.
        angle: number from 1 to 360.
        lst_imgs: list of image strings.
    OUTPUT
        Rotated Images
    '''

    for l in lst_imgs:
        img = io.imread(file_path + str(l) + '.jpeg')
        img = rotate(img, angle)
        io.imsave(file_path + str(l) + '_' + str(angle) + '.jpeg', img)


def mirror_images(file_path, imgs):
    '''
    Flip and mirror image
    INPUT
        file_path: Images path
        imgs: list of image strings.
    OUTPUT
        Images mirrored and fliped
    '''

    for l in imgs:
        img = cv2.imread(file_path + str(l) + '.jpeg')
        img = cv2.flip(img, 1)
        cv2.imwrite(file_path + str(l) + '_mir' + '.jpeg', img)


def convert_images_to_arrays_train(file_path, df):
    """
    Convert path images to npfile
    INPUT
        file_path: Specified file path for resized test and train images.
        df: Pandas DataFrame being used to assist file imports.
    OUTPUT
        NumPy array of image arrays.
    """

    lst_imgs = [l for l in df['train_image_name']]

    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs])


def create_directory(directory):
    '''
    Create folder
    INPUT
        directory: path, called by "folder/"
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

def crop_and_resize_images(path, new_path, cropx, cropy, img_size=256):
    '''
    Crops, resizes, and stores all images from a directory in a new directory.
    INPUT
        path: path to the original images
        new_path: Path to save resized images
        img_size: New size for the rescaled images.
    '''
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store'] # MACOS
    total = 0
    last = 0
    tam = len(dirs)
    print("Resized 0%/100%")
    for item in dirs:
        img = io.imread(path+item)
        y,x,channel = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[starty:starty+cropy,startx:startx+cropx]
        img = resize(img, (256,256))
        io.imsave(str(new_path + item), img)
        total += 1
        percent =(total/tam)*100 
        if(percent- last >= 10 or percent > 99):
            print("Resized ",percent,"/100%")
            last = percent

        #print("Saving: ", item, total)

def get_lst_images(file_path):
    """
    Get List images
    INPUT
        file_path: path
    OUTPUT
        List of image strings
    """
    return [i for i in os.listdir(file_path) if i != '.DS_Store']

# ---------------------- DATA FUNCTIONS -------------------------

def split_input_data(X, y, test_data_size):
    """
    Split data into test and training datasets.
    INPUT
        X: Data
        y: labels
        test_data_size: percent of test data
    OUPUT
        arrays: X_train, X_test, y_train, and y_test
    """
    return train_test_split(X, y, test_size=test_data_size, random_state=42)


def reshape_data(arr, img_rows, img_cols, channels):
    """
    Reshapes the data into network format
    INPUT
        arr: Images array.
        img_rows: height
        img_cols: width
        channels: grayscale (1) or RGB (3)
    OUTPUT
        Reshaped array of NumPy arrays.
    """
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


# ------------------------- CNN FUNCTIONS ---------------------------

def cnn_model_1(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, nb_gpus):
    """
    Define and run the Convolutional Neural Network
    INPUT
        X_train: Train data
        X_test: Test data
        y_train: Train label
        y_test: Test label
        kernel_size: Initial size of kernel
        nb_filters: Initial number of filters
        channels: grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification
    OUTPUT
        Fitted CNN model
    """

    model = Sequential()

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(img_rows, img_cols, channels), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #model = multi_gpu_model(model, gpus=nb_gpus)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    stop = EarlyStopping(monitor='val_acc',
                         min_delta=0.001,
                         patience=2,
                         verbose=0,
                         mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1,
              validation_split=0.2,
              class_weight='auto',
              callbacks=[stop, tensor_board])

    return model

def cnn_model(X_train, X_test, y_train, y_test, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, weights):
    """
    Define and run the Convolutional Neural Network
    INPUT
        X_train: Train data
        X_test: Test data
        y_train: Train label
        y_test: Test label
        kernel_size: Initial size of kernel
        nb_filters: Initial number of filters
        channels: grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification
    OUTPUT
        Fitted CNN model
    """
    model = Sequential()


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
        padding='valid',
        strides=4,
        input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))


    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2,2)))


    kernel_size = (16,16)
    model.add(Conv2D(64, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))


    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)


    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


    model.compile(loss = 'categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


    stop = EarlyStopping(monitor='val_acc',
                            min_delta=0.001,
                            patience=2,
                            verbose=0,
                            mode='auto')


    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


    model.fit(X_train,y_train, batch_size=batch_size, epochs=nb_epoch,
                verbose=1,
                validation_split=0.2,
                class_weight=weights,
                callbacks=[stop, tensor_board])

    return model


def save_model(model, score, model_name):
    """
    Saves Keras model to an h5 file, based on precision_score
    INPUT
        model: Keras model object to be saved
        score: Score to determine if model should be saved.
        model_name: name of model to be saved
    """

    if score >= 0.75:

        print("Saving Model")
        model.save("../models/" + model_name + "_recall_" + str(round(score, 4)) + ".h5")
    else:
        print("Model Not Saved.  Score: ", score)


# --------- PROCESSING --------

def CleanDatabase():
    """
        Clean Black Images Database
    """
    start_time = time.time()
    tLabels = pd.read_csv(labels)

    #print(tLabels)
    tLabels['image'] = [i + '.jpeg' for i in tLabels['image']]
    tLabels['black'] = np.nan

    tLabels['black'] = find_black_images(folder_train, tLabels)
    tLabels = tLabels.loc[tLabels['black'] == 0]
    tLabels.to_csv(trainLabels, index=False, header=True)

    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))


def ResizeImages():
    """
        Crop resize and pre processing images
    """
    
    crop_and_resize_images(path=folder_train, new_path=folder_train_resized, cropx=1800, cropy=1800, img_size=256)
    #crop_and_resize_images(path=folder_test, new_path=folder_test_resized, cropx=1800, cropy=1800, img_size=256)

def GeneratePlasticDatabase():
    """
        Generate plastic database to potentialize the learning process
    """
    print("Generating plastic database")
    start_time = time.time()
    tLabels = pd.read_csv(trainLabels)

    tLabels['image'] = tLabels['image'].str.rstrip('.jpeg')
    tLabels_no_DR = tLabels[tLabels['level'] == 0] # Note: 0 DR dont`t have Diabete Retinopatica
    tLabels_DR = tLabels[tLabels['level'] >= 1]

    lst_imgs_no_DR = [i for i in tLabels_no_DR['image']]
    lst_imgs_DR = [i for i in tLabels_DR['image']]

    # Mirror Images with no DR
    print("Mirroring Non-DR Images")
    mirror_images(folder_train_resized, lst_imgs_no_DR)


    # Rotate all images that have any level of DR - this is for generate plastic dataset
    print("Rotating 90")
    rotate_images(folder_train_resized, 90, lst_imgs_DR)

    print("Rotating 120")
    rotate_images(folder_train_resized, 120, lst_imgs_DR)

    print("Rotating 180")
    rotate_images(folder_train_resized, 180, lst_imgs_DR)

    print("Rotating 270")
    rotate_images(folder_train_resized, 270, lst_imgs_DR)

    print("Mirroring DR Images")
    mirror_images(folder_train_resized, lst_imgs_DR)

    print("Generating Plastic database Completed")
    print("--- %s seconds ---" % (time.time() - start_time))

    pass


def PreLoad():
    """
        Preload file images and generate a unique file
    """
    start_time = time.time()

    labels = pd.read_csv(customTrainLabels)

    print("Preloading images")
    X_train = convert_images_to_arrays_train(folder_train_resized, labels)

    print(X_train.shape)

    print("Saving in file")
    np.save(file_X_npy, X_train)

    print("--- %s seconds ---" % (time.time() - start_time))

def MakeTrainLabel():
    """
        Make custom label to train CNN
    """
    tLabels = pd.read_csv(trainLabels)

    lst_imgs = get_lst_images(folder_train_resized)

    new_tLabels = pd.DataFrame({'image': lst_imgs})
    new_tLabels['image2'] = new_tLabels.image

    # Remove suffix
    new_tLabels['image2'] = new_tLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

    # Remove .jpeg
    new_tLabels['image2'] = new_tLabels.loc[:, 'image2'].apply(
        lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg')+".jpeg")

    # tLabels = tLabels[0:10]
    new_tLabels.columns = ['train_image_name', 'image']
    
    tLabels = pd.merge(tLabels, new_tLabels, how='outer', on='image')
    tLabels.drop(['black'], axis=1, inplace=True)

    print(tLabels.shape)
    tLabels = tLabels.dropna()
    print(tLabels.shape)

    print("Writing CSV")
    tLabels.to_csv(customTrainLabels, index=False, header=True)

def PreProcess():
    CleanDatabase()
    ResizeImages()
    GeneratePlasticDatabase()
    MakeTrainLabel()
    PreLoad()


def Train():
    # Import data
    labels = pd.read_csv(customTrainLabels)
    X = np.load(file_X_npy)
    y = np.array(labels['level'])

    # Class Weights (for imbalanced classes)
    print("Computing Class Weights")
    weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

    print("Splitting data into test/ train datasets")
    X_train, X_test, y_train, y_test = split_input_data(X, y, 0.2)


    print("Reshaping Data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)

    print("X_train Shape: ", X_train.shape)
    print("X_test Shape: ", X_test.shape)


    input_shape = (img_rows, img_cols, channels)


    print("Normalizing Data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255


    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print("y_train Shape: ", y_train.shape)
    print("y_test Shape: ", y_test.shape)


    print("Training Model")
    
    model = cnn_model(X_train, X_test, y_train, y_test, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes,weights)
    #model = cnn_model_1(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, nb_gpus=0)

    print("Predicting")
    y_pred = model.predict(X_test)


    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


    y_pred = [np.argmax(y) for y in y_pred]
    y_test = [np.argmax(y) for y in y_test]


    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')


    print("Precision: ", precision)
    print("Recall: ", recall)
    save_model(model=model, score=recall, model_name="DR_Two_Classes")


PreProcess()

#Train()