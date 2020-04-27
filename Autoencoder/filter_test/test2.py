from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import keras
import matplotlib.pyplot as plt
import tensorflow as tf 
import cv2
import os
from keras.layers.advanced_activations import LeakyReLU



input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

initial_mode = keras.initializers.lecun_normal(seed=None)
initial_mode2 = keras.initializers.lecun_normal(seed=None)


activation_method = LeakyReLU(alpha=0.2)




for i in range(1,2):
    ## Build the Convolution autoencoders model
    
    x = Conv2D(16, (3, 3), activation=activation_method, padding='same',kernel_initializer=initial_mode)(input_img) # 28*28*16
    x = MaxPooling2D((2, 2), padding='same')(x)  # 14*14*16
    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(x) # 14*14*8
    x = MaxPooling2D((2, 2), padding='same')(x) # 7*7*8
    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(x) # 7*7*8
    encoded = MaxPooling2D((2, 2), padding='same')(x) # 4*4*8

    # This part is full connect layer
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(encoded) # 4*4*8
    x = UpSampling2D((2, 2))(x) # 8*8*8
    x = Conv2D(8, (3, 3), activation='relu', padding='same',kernel_initializer=initial_mode)(x) # 8*8*8
    x = UpSampling2D((2, 2))(x) # 16*16*8
    x = Conv2D(16, (3, 3), activation='relu',kernel_initializer=initial_mode)(x) # 14*14*16
    x = UpSampling2D((2, 2))(x) # 28*28*16
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',kernel_initializer=initial_mode2)(x) # 28*28*1
    


    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')

    from keras.datasets import mnist
    import numpy as np

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    from keras.callbacks import TensorBoard

    
    

    class CustomSaver(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs={}):            
            loop = epoch + 1
            
            if epoch ==0:
                print('first epoch')

            
            elif loop%5 == 0:  # or save after some epoch, each k-th epoch etc.
                print('Save the image')
                decoded_imgs = autoencoder.predict(x_test)
                img = decoded_imgs[0]
                img = img.reshape(28,28)
                
                # save the img
                current_dir = os.getcwd()
                plt.imshow(img,'gray')
                plt.savefig(current_dir+'/Decoded_img/Decoded_image_epoch'+str(loop)+'.png')
                
                
                

    saver = CustomSaver()
    current_dir = os.getcwd()

    result = autoencoder.fit(x_train, x_train,
                                        epochs=50,
                                        batch_size=128,
                                        shuffle=True,
                                        validation_data=(x_test, x_test),
                                        callbacks=[saver,TensorBoard(log_dir=current_dir+'/tensorboard_data/test_i_g_a')])
    
    


#print(autoencoder.summary())




#decoded_images = autoencoder.predict(x_test)