# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:32:39 2024

@author: user
"""

from vit_keras import vit
from tensorflow.keras import backend as K
from tensorflow.keras import models, metrics, callbacks, layers, losses, preprocessing
from attention_DFU_utils import plot_training, extract_bboxes, heatmap_and_mask
import tensorflow_addons as tfa

dtype='float32'
K.set_floatx(dtype)

#%%
""" Initialize parameters """

IMAGE_SIZE = (224,224)
BATCH_SIZE = 12
EPOCHS = 20
TRAIN_PATH = "C:/Users/user/Desktop/DFU_attentions/train_on_synth"


#%%
""" Data Generator """

datagen = preprocessing.image.ImageDataGenerator(
    rescale = 1/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.25)


train_gen=datagen.flow_from_directory(
    directory=TRAIN_PATH,
    subset="training",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="binary",
    target_size=IMAGE_SIZE,
)

val_gen=datagen.flow_from_directory(
    directory=TRAIN_PATH,
    subset="validation",
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=False,
    class_mode="binary",
    target_size=IMAGE_SIZE)

#%% 
""" Setup """

learning_rate = 1e-4
#should be 1e-3 for no fine tuning

optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)

reduce_lr = callbacks.ReduceLROnPlateau(monitor = 'val_acc',
                                                 factor = 0.1,
                                                 patience = 5,
                                                 verbose = 1,
                                                 min_delta = 1e-4,
                                                 min_lr = 1e-6,
                                                 mode = 'max')

earlystopping = callbacks.EarlyStopping(monitor = 'val_acc',
                                                 min_delta = 1e-4,
                                                 patience = 10,
                                                 mode = 'max',
                                                 restore_best_weights = True,
                                                 verbose = 1)

checkpointer =callbacks.ModelCheckpoint(filepath = './model.hdf5',
                                                  monitor = 'val_acc',
                                                  verbose = 1,
                                                  save_best_only = True,
                                                  save_weights_only = True,
                                                  mode = 'max')
# 
callbacks = [earlystopping, checkpointer, reduce_lr]

acc = []
val_acc = []

loss = []
val_loss = []

#%% 
""" Training the model """

for i in range (1):
    K.clear_session()
    
    vit_model = vit.vit_b16(
            image_size = IMAGE_SIZE,
            activation = 'sigmoid',
            pretrained = True,
            include_top = False,
            pretrained_top = False)
    
        
    model = models.Sequential([
            vit_model,
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(16, activation = tfa.activations.gelu),
            layers.Dense(1, 'sigmoid')
        ],
        name = 'vision_transformer')    


    model.compile(optimizer = optimizer,
                  loss = losses.BinaryCrossentropy(label_smoothing = 0.2),
                  metrics=[metrics.BinaryAccuracy(name = 'acc')])

    history = model.fit(x = train_gen,
              validation_data = val_gen,
              epochs = EPOCHS,
              callbacks = callbacks)
    
    acc.append(history.history['acc'])
    val_acc.append(history.history['val_acc'])
    
    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    
    
#%%
""" Plot training plots """

data_hist = [acc, loss]
val_data_hist = [val_acc, val_loss]

plot_training(data_hist, val_data_hist)

#%%

# model = models.load_model("dfu.h5")


#%%
""" Attention """


IMG_PATH = "F:/DFU/DFUC2022_train/DFUC2022_train_release/DFUC2022_train_images/"
#IMG_PATH = "C:/Users/user/Downloads/DFUC2024_test_release/DFUC2024_test_release/"
BBOX_FOLDER_IMG = "C:/Users/user/Desktop/DFU_attentions/bboxes/patches/patch/"
BBOX_FOLDER_MASK = "C:/Users/user/Desktop/DFU_attentions/bboxes/patches/mask/"
VIS_FOLDER = "C:/Users/user/Desktop/DFU_attentions/bboxes/visualisation/"

extract_bboxes(IMG_PATH, BBOX_FOLDER_IMG, BBOX_FOLDER_MASK, VIS_FOLDER, model)


#%%
""" Masks and attention dla Darka """

IMG_PATH = "C:/Users/user/Downloads/DFUC2024_val_release/DFUC2024_val_release/"
#IMG_PATH = "C:/Users/user/Downloads/DFUC2024_train_release/DFUC2024_train_release/Synthetic_images/"
OUT_HEAT_PATH = "C:/Users/user/Desktop/DFU_attentions/dlaDarka2/heatmaps/oryginal/"
OUT_MASK_PATH = "C:/Users/user/Desktop/DFU_attentions/val_masks_imagenet/"


heatmap_and_mask(IMG_PATH, OUT_HEAT_PATH, OUT_MASK_PATH, model)



