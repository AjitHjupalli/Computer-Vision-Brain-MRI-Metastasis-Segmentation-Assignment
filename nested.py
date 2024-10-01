from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def create_nested_unet(input_shape=(256, 256, 1)):
    input_layer = Input(input_shape)

    return Model(input_layer, outputs)
