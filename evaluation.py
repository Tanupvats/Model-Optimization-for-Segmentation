
import tensorflow as tf
import time
import os
import numpy as np
from data_preparation import load_datasets
from utils import get_gzipped_model_size

def evaluate_model_size():
    original_model_size = get_gzipped_model_size('models/original_model.h5')
    pruned_model_size = get_gzipped_model_size('models/pruned_model.h5')
    dynamic_quantized_size = os.path.getsize('models/dynamic_quantized_model.tflite')
    int8_quantized_size = os.path.getsize('models/int8_quantized_model.tflite')
    float16_quantized_size = os.path.getsize('models/float16_quantized_model.tflite')

    print("\nModel Sizes:")
    print(f"Original Model Size: {original_model_size / 1024:.2f} KB")
    print(f"Pruned Model Size: {pruned_model_size / 1024:.2f} KB")
    print(f"Dynamic Quantized Model Size: {dynamic_quantized_size / 1024:.2f} KB")
    print(f"Int8 Quantized Model Size: {int8_quantized_size / 1024:.2f} KB")
    print(f"Float16 Quantized Model Size: {float16_quantized_size / 1024:.2f} KB")

def evaluate_inference_time(interpreter, test_dataset, num_images=100):
    total_time = 0
    for i, (input_image, _) in enumerate(test_dataset.take(num_images)):
        input_image = input_image.numpy()
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_image)
        start_time = time.time()
        interpreter.invoke()
        total_time += time.time() - start_time
    avg_time = total_time / num_images
    return avg_time

def evaluate_model_accuracy(interpreter, dataset):
    OUTPUT_CHANNELS = 3
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=OUTPUT_CHANNELS)
    for input_image, target_mask in dataset.take(100):
        input_image = input_image.numpy()
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_image)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        predicted_mask = tf.argmax(output, axis=-1)
        iou_metric.update_state(target_mask.numpy().flatten(), predicted_mask.numpy().flatten())
    return iou_metric.result().numpy()

def evaluate_models():
    _, val_dataset = load_datasets()
    test_dataset = val_dataset.unbatch().batch(1)

    # Evaluate Dynamic Range Quantized Model
    interpreter_dynamic = tf.lite.Interpreter(model_path='models/dynamic_quantized_model.tflite')
    interpreter_dynamic.allocate_tensors()
    dynamic_inference_time = evaluate_inference_time(interpreter_dynamic, test_dataset)
    dynamic_accuracy = evaluate_model_accuracy(interpreter_dynamic, test_dataset)
    print(f"\nDynamic Range Quantized Model - Inference Time: {dynamic_inference_time:.4f}s, Accuracy (Mean IoU): {dynamic_accuracy:.4f}")

    # Evaluate Full Integer Quantized Model
    interpreter_int8 = tf.lite.Interpreter(model_path='models/int8_quantized_model.tflite')
    interpreter_int8.allocate_tensors()
    int8_inference_time = evaluate_inference_time(interpreter_int8, test_dataset)
    int8_accuracy = evaluate_model_accuracy(interpreter_int8, test_dataset)
    print(f"Int8 Quantized Model - Inference Time: {int8_inference_time:.4f}s, Accuracy (Mean IoU): {int8_accuracy:.4f}")

    # Evaluate Float16 Quantized Model
    interpreter_float16 = tf.lite.Interpreter(model_path='models/float16_quantized_model.tflite')
    interpreter_float16.allocate_tensors()
    float16_inference_time = evaluate_inference_time(interpreter_float16, test_dataset)
    float16_accuracy = evaluate_model_accuracy(interpreter_float16, test_dataset)
    print(f"Float16 Quantized Model - Inference Time: {float16_inference_time:.4f}s, Accuracy (Mean IoU): {float16_accuracy:.4f}")

def main():
    evaluate_model_size()
    evaluate_models()

if __name__ == "__main__":
    main()
