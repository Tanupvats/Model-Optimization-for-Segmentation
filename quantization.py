
import tensorflow as tf
from data_preparation import load_datasets
from pruning import unstructured_pruning
from training import train_model

def representative_data_gen():
    train_dataset, _ = load_datasets()
    for input_value, _ in train_dataset.take(100):
        yield [input_value.numpy()]

def dynamic_range_quantization(pruned_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_dynamic = converter.convert()
    with open('models/dynamic_quantized_model.tflite', 'wb') as f:
        f.write(tflite_model_dynamic)
    print("Dynamic range quantized model saved at 'models/dynamic_quantized_model.tflite'")

def int8_quantization(pruned_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8
    tflite_model_int8 = converter.convert()
    with open('models/int8_quantized_model.tflite', 'wb') as f:
        f.write(tflite_model_int8)
    print("Int8 quantized model saved at 'models/int8_quantized_model.tflite'")

def float16_quantization(pruned_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_float16 = converter.convert()
    with open('models/float16_quantized_model.tflite', 'wb') as f:
        f.write(tflite_model_float16)
    print("Float16 quantized model saved at 'models/float16_quantized_model.tflite'")

if __name__ == "__main__":
    # Ensure pruned model is available
    pruned_model = tf.keras.models.load_model('models/pruned_model.h5')
    dynamic_range_quantization(pruned_model)
    int8_quantization(pruned_model)
    float16_quantization(pruned_model)
