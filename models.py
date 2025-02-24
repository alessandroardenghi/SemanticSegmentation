import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

def build_base_model(input_size=(224, 224, 4)):
    # Base Model
    model = tf.keras.Sequential()
    model.add(Input(shape=input_size))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(6, kernel_size=(3, 3), activation='softmax', padding='same'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_shallow_unet(input_size=(224, 224, 5)):
    
    # Encoder Decoder Model (Shallow Unet)
    input = Input(input_size)

    # ENCODER 
    first_conv = Conv2D(32, kernel_size = (3, 3), 
                        activation = 'relu', padding = 'same')(input)                                           # (224, 224, 5) -> (224, 224, 32)
    first_pool = MaxPooling2D(pool_size = (2, 2))(first_conv)                                                   # (224, 224, 32) -> (112, 112, 32)

    second_conv = Conv2D(64, kernel_size = (3, 3), 
                         activation = 'relu', padding = 'same')(first_pool)                                     # (112, 112, 32) -> (112, 112, 64)
    second_pool = MaxPooling2D(pool_size = (2, 2))(second_conv)                                                 # (112, 112, 64) -> (56, 56, 64)

    # BOTTLENECK
    third_conv = Conv2D(64, kernel_size = (3, 3), 
                        activation = 'relu', padding = 'same')(second_pool)                                     # (56, 56, 64) -> (56, 56, 64)
    first_upsample = Conv2DTranspose(64, kernel_size = (3, 3), 
                                     activation = 'relu', strides = (2, 2), padding = 'same')(third_conv)       # (56, 56, 64) -> (112, 112, 64)

    # DECODER 
    concat_1 = Concatenate()([first_upsample, second_conv])                                                     # (112, 112, 64) + (112, 112, 64) -> (112, 112, 128)
    second_upsample = Conv2DTranspose(32, kernel_size= (3, 3), 
                                      activation = 'relu', strides = (2, 2), padding = 'same')(concat_1)        # (112, 112, 128) -> (224, 224, 32)

    concat_1 = Concatenate()([second_upsample, first_conv])                                                     # (224, 224, 32)  + (224, 224, 32) -> (224, 224, 64)
    third_upsample = Conv2DTranspose(32, kernel_size = (3, 3), 
                                     activation = 'relu', padding = 'same')(second_upsample)                    #(224, 224, 64) -> (224, 224, 32)


    output = Conv2D(6, kernel_size = (3, 3), activation = 'softmax', padding = 'same')(third_upsample)

    model = Model(inputs=input, outputs=output)
    
    return model
