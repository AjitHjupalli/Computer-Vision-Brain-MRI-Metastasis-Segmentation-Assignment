from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_augmenters(images, masks):
    augmentation_params = {
        'rotation_range': 30,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.2,
        'horizontal_flip': True
    }
    
    img_datagen = ImageDataGenerator(**augmentation_params)
    mask_datagen = ImageDataGenerator(**augmentation_params)

    return img_datagen, mask_datagen
