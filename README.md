# Google-Image-Classifier-Project-using-Keras-Tensorflow
Google Image Classifier Project using Keras &amp; Tensorflow

# Image Classifier Project

This repository contains the implementation for an image classification project using TensorFlow and TensorFlow Hub. The goal of this project is to classify images of flowers into different categories using a pre-trained model (MobileNet V2) for feature extraction.

## Project Overview

The project is divided into the following sections:

1. **Data Loading and Preprocessing**: Loading the Oxford Flowers 102 dataset and applying necessary transformations.
2. **Model Building and Training**: Constructing and training a neural network using MobileNet V2 for feature extraction.
3. **Model Evaluation**: Evaluating the trained model on the test dataset and plotting accuracy and loss.
4. **Prediction Script**: Predicting flower classes from input images using the trained model.

## Files in the Repository

- `train.py`: Script for training the image classifier model.
- `predict.py`: Script for predicting flower classes from an input image using a saved Keras model.
- `label_map.json`: JSON file containing label-to-flower name mappings.
- `flower_classifier.h5`: Saved Keras model file.
- `Project_Image_Classifier_Project.ipynb`: Jupyter notebook for the entire workflow.
- `README.md`: This file.

## Requirements

- TensorFlow
- TensorFlow Hub
- TensorFlow Datasets
- Matplotlib
- Pillow (PIL)
- NumPy

You can install the required packages using:

```bash
pip install tensorflow tensorflow_hub tensorflow_datasets matplotlib pillow numpy
```

## Usage

- Training the Model
- To train the model, run the train.py script:

```python train.py```

- This script will:
  1. Load the Oxford Flowers 102 dataset.
  2. Preprocess the images.
  3. Build and train a neural network using MobileNet V2 for feature extraction.
  4. Save the trained model to flower_classifier.h5.
 
## Evaluating the Model

- To evaluate the model on the test dataset, include the evaluation section in your script or Jupyter notebook:
```# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_batches)
print(f"Test accuracy: {test_accuracy}")
```

## Plotting Accuracy and Loss

- Include the following code to plot training and validation accuracy and loss:

```# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
```

## Making Predictions

- To predict flower classes from an image, run the predict.py script:

```python predict.py /path/to/image /path/to/flower_classifier.h5 --category_names label_map.json --top_k 5```

- This script will:
  1. Load the trained Keras model.
  2. Preprocess the input image.
  3. Predict the top K flower classes.
  4. Print the predicted classes and their probabilities.
 
## Acknowledgements

- This project uses the Oxford Flowers 102 dataset.
- The feature extraction is done using the MobileNet V2 model from TensorFlow Hub.
