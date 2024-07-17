# Image Classifier Project - Part 2

This repository contains the implementation for Part 2 of the Image Classifier project using TensorFlow. Below are the details for each file included:

### Files Included:

1. **predict.py**: Python script for predicting flower classes from an image using a pre-trained Keras model. It takes an image path and model path as input and optionally returns the top K classes along with probabilities.

### Usage:

To predict the flower class from an image, run the `predict.py` script with the following command:


- `image_path`: Path to the image you want to classify.
- `model_path`: Path to the saved Keras model (`flower_classifier.h5`).
- `--top_k`: Number of top classes to return (default is 5).
- `--category_names`: Optional path to a JSON file mapping labels to category names.

Make sure you have TensorFlow and other required dependencies installed.

### Requirements:

- TensorFlow
- TensorFlow Hub
- NumPy
- PIL (Pillow)

### Notes:

- Ensure the model (`flower_classifier.h5`) and label mapping file (`label_map.json`) are in the same directory as `predict.py` for correct execution.

For more details, refer to the project rubric and guidelines.
