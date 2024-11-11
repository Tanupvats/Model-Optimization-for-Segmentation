

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 128
BATCH_SIZE = 32
BUFFER_SIZE = 1000

def preprocess_data(sample):
    image = tf.image.resize(sample['image'], (IMG_SIZE, IMG_SIZE))
    mask = tf.image.resize(sample['segmentation_mask'], (IMG_SIZE, IMG_SIZE))
    
    # Normalize the images
    image = image / 255.0
    mask -= 1  # Adjust mask values to start from 0
    
    return image, mask

def load_datasets():
    dataset, info = tfds.load('oxford_iiit_pet', with_info=True)
    train_dataset = dataset['train'].map(preprocess_data).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = dataset['test'].map(preprocess_data).batch(BATCH_SIZE)
    return train_dataset, val_dataset
