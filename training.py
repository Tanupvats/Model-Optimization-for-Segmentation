
import tensorflow as tf
from model_definition import unet_model, OUTPUT_CHANNELS
from data_preparation import load_datasets

def train_model():
    train_dataset, val_dataset = load_datasets()
    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    EPOCHS = 5  # Adjust as needed
    model.fit(train_dataset,
              epochs=EPOCHS,
              validation_data=val_dataset)
    # Save the original model
    model.save('models/original_model.h5')
    print("Original model saved at 'models/original_model.h5'")
    return model, train_dataset, val_dataset

if __name__ == "__main__":
    train_model()
