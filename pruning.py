
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from training import train_model

def unstructured_pruning(model, train_dataset, val_dataset):
    # Calculate the number of training steps
    EPOCHS = 5  # Should match the training epochs
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    end_step = EPOCHS * steps_per_epoch

    # Define pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=end_step)
    }

    # Apply pruning
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Compile the model
    pruned_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    # Define pruning callbacks
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Fine-tune the pruned model
    pruned_model.fit(train_dataset,
                     epochs=EPOCHS,
                     validation_data=val_dataset,
                     callbacks=callbacks)

    # Strip pruning wrappers
    pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    pruned_model.save('models/pruned_model.h5')
    print("Pruned model saved at 'models/pruned_model.h5'")
    return pruned_model

if __name__ == "__main__":
    # Load original model and datasets
    original_model, train_dataset, val_dataset = train_model()
    unstructured_pruning(original_model, train_dataset, val_dataset)
