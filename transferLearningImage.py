import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import tensorflow.keras.metrics
from tensorflow.keras import layers
#import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def trainmodel(data_dir, img_width, img_height, modelname, batch_size, initial_epochs, fine_tune_epochs,  base_model):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      labels='inferred',
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size,
      label_mode = 'binary')

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      labels='inferred',
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size, label_mode = 'binary')
    
    '''
    class_names = train_ds.class_names
    file_paths_train = train_ds.file_paths
    file_paths_val = val_ds.file_paths
    intersection_set = set.intersection(set(file_paths_train), set(file_paths_val))
    intersection_list = list(intersection_set)
    class_names = np.array(train_ds.class_names)
    print(class_names)
    '''
    
    AUTOTUNE = tf.data.AUTOTUNE    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    #data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])
  
    base_model = base_model
    
    base_model.trainable = False
        
    inputs = keras.Input(shape=(224,224,3))
    #x = data_augmentation(inputs) 
    scale_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.summary()
   

    model.compile(
    optimizer= keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()])
    
    print('Transfer Learning is starting')
    
    filepath= '/home/apps/computerVision/imageClassification/terrorismClassification/models/' + modelname + '-'+ 'TopLayer-' +'weights.{epoch:02d}-{loss:.2f}-{binary_accuracy:.2f}-{val_binary_accuracy:.2f}.h5'
    #filepath= '/Users/irvanseptiar/MKI/CV/terrorismClassification/' + modelname + '-'+ 'TopLayer-' +'weights.{epoch:02d}-{loss:.2f}-{binary_accuracy:.2f}-{val_loss:.2f}-{val_binary_accuracy:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    history = model.fit(train_ds, epochs=initial_epochs, batch_size=batch_size,validation_data=(val_ds), callbacks=[checkpoint])
    print('Transfer Learning is Done')
    
    print()
    print('Fine Tunning Step is starting')
    base_model.trainable = True
    model.summary()

    model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()])
    
    #filepathTune= '/Users/irvanseptiar/MKI/CV/terrorismClassification/' + modelname + '-'+ 'AllLayer-' +'weights.{epoch:02d}-{loss:.2f}-{binary_accuracy:.2f}-{val_loss:.2f}-{val_binary_accuracy:.2f}.h5'
    filepathTune= '/home/apps/computerVision/imageClassification/terrorismClassification/models/' + modelname + '-'+ 'AllLayer-' +'weights.{epoch:02d}-{loss:.2f}-{binary_accuracy:.2f}-{val_binary_accuracy:.2f}.h5'
    checkpointTune = ModelCheckpoint(filepathTune, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
    historyfinetune = model.fit(train_ds, epochs= initial_epochs + fine_tune_epochs, \
                                initial_epoch=history.epoch[-1],\
                                    batch_size=batch_size,validation_data=(val_ds), callbacks=[checkpointTune])
    
    print('Fine Tunning Step is Done')
    return model, history, historyfinetune

if __name__ == '__main__':
    #base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model = keras.applications.Xception(weights="imagenet", include_top=False, input_shape=(224,224,3))
    data_dir = '/home/apps/computerVision/imageClassification/terrorismClassification/dataset/'
    img_width, img_height, batch_size, initial_epochs, fine_tune_epochs = 224, 224, 32, 10, 10
    trainmodel(data_dir = data_dir, img_width = img_width, img_height = img_height,  modelname = 'terrorism', batch_size = batch_size, initial_epochs = initial_epochs, fine_tune_epochs = fine_tune_epochs, base_model = base_model)

