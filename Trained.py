from models.nested_unet import nested_unet
from preprocess.clahe_preprocess import preprocess_images
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

data_directory = 'data/'
image_data, mask_data = preprocess_images(data_directory)

model = nested_unet()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_callback = ModelCheckpoint('weights/nested_unet_weights.h5', monitor='val_loss', save_best_only=True)

model.fit(image_data, mask_data, validation_data=(test_images, test_masks), epochs=50, callbacks=[checkpoint_callback])
