import tensorflow as tf
import numpy as np
import argparse
import json
from PIL import Image

# Function to process image
def process_image(image):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image.numpy()

# Function to predict
def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    probs, classes = tf.math.top_k(predictions, k=top_k)
    return probs.numpy()[0], classes.numpy()[0]

# Function to load label mapping
def load_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained deep learning model.')
    parser.add_argument('image_path', action="store", help='Path to the image')
    parser.add_argument('model', action="store", help='Path to the saved Keras model')
    parser.add_argument('--top_k', action="store", type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', action="store", help='Path to a JSON file mapping labels to flower names')
    
    args = parser.parse_args()

    # Load the Keras model
    model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer': hub.KerasLayer})

    # Load class names if provided
    if args.category_names:
        class_names = load_class_names(args.category_names)
    else:
        class_names = None

    # Predict flower name
    probs, classes = predict(args.image_path, model, args.top_k)

    # Print top K classes and probabilities
    for i in range(args.top_k):
        class_label = classes[i]
        class_name = class_names[str(class_label)] if class_names else class_label
        print(f"Predicted class: {class_name}, Probability: {probs[i]}")

if __name__ == '__main__':
    main()
