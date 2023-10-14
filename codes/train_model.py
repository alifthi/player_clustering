import tensorflow as tf
from keras import layers as ksl
from glob import glob
import cv2 as cv
import numpy as np
from config import SAVE_MODEL_PATH, INPUT_SHAPE, EMBEDDING_DIM,\
                    LEARNING_RATE, LOSS, METRICS,\
                    EPOCHS, BATCH_SIZE, VALIDATION_SPLIT
                    

class train_model:
    def build_model(self):
        encoderInput = ksl.Input(INPUT_SHAPE)

        x = ksl.Conv2D(16,kernel_size=3,padding='same',activation='gelu')(encoderInput)
        x = ksl.Conv2D(16,kernel_size=3,padding='same',activation='gelu')(x)
        x = ksl.BatchNormalization()(x)
        x = ksl.AveragePooling2D(2)(x)

        x = ksl.Conv2D(8,kernel_size=3,padding='same',activation='gelu')(x)
        x = ksl.Conv2D(8,kernel_size=3,padding='same',activation='gelu')(x)
        x = ksl.BatchNormalization()(x)
        x = ksl.AveragePooling2D(2)(x)


        x = ksl.Conv2D(EMBEDDING_DIM,kernel_size=3,padding='same',activation='gelu')(x)
        x = ksl.Conv2D(EMBEDDING_DIM,kernel_size=3,padding='same',activation='gelu')(x)
        x = ksl.BatchNormalization()(x)
        x = ksl.AveragePooling2D(2)(x)

        self.encoder = tf.keras.Model(encoderInput,x)
        decoderInput = ksl.Input([8,8,EMBEDDING_DIM])
        x = ksl.Conv2DTranspose(8,kernel_size=3,strides=2,padding='same',activation='gelu')(decoderInput)
        x = ksl.BatchNormalization()(x)

        x = ksl.Conv2DTranspose(16,kernel_size=3,strides=2,padding='same',activation='gelu')(x)
        x = ksl.BatchNormalization()(x)

        x = ksl.Conv2DTranspose(3,kernel_size=3,strides=2,padding='same',activation='sigmoid')(x)

        decoder = tf.keras.Model(decoderInput,x)
        self.AEModel = tf.keras.Sequential([self.encoder,decoder])
    def compile_model(self):
        optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.AEModel.compile(loss=LOSS,optimizer=optim,metrics=METRICS)
    def train_model(self,images):
        self.AEModel.fit(images,images,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_split=VALIDATION_SPLIT)
    def save_model(self):
        self.AEModel.save(SAVE_MODEL_PATH+'AEModel.h5')
        self.encoder.save(SAVE_MODEL_PATH+'Encoder.h5')
    def load_model(self):
        self.AEModel = tf.keras.models.load_model(SAVE_MODEL_PATH+'AEModel.h5')
        self.encoder = tf.keras.models.load_model(SAVE_MODEL_PATH+'Encoder.h5')