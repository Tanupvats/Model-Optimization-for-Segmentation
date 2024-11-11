import tensorflow as tf

IMG_SIZE = 128
OUTPUT_CHANNELS = 3  # Number of classes in the segmentation mask

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Encoder
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D()(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D()(conv2)
    
    # Bottleneck
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    
    # Decoder
    up1 = tf.keras.layers.UpSampling2D()(conv3)
    concat1 = tf.keras.layers.Concatenate()([up1, conv2])
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(concat1)
    
    up2 = tf.keras.layers.UpSampling2D()(conv4)
    concat2 = tf.keras.layers.Concatenate()([up2, conv1])
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(concat2)
    
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(conv5)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
