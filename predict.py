import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image


def process_image(image):
    """
    Takes in a NumPy image array and returns a processed image
    as a NumPy array with shape (224, 224, 3).
    """
    img_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, (224, 224))
    img_tensor = img_tensor / 255.0
    return img_tensor.numpy()


def predict(image_path, model, top_k=5):
    """
    Predict the top K most probable classes for an image.

    Returns:
        probs   -> Top K probability values (1D NumPy array)
        classes -> Top K class indices as strings (e.g. ['70', '3', ...])
    """
    # Load image
    im = Image.open(image_path)
    np_image = np.asarray(im)

    # Preprocess
    processed_image = process_image(np_image)

    # Add batch dimension
    input_tensor = np.expand_dims(processed_image, axis=0)

    # Predict
    preds = model.predict(input_tensor)

    # Top K
    top_probs, top_indices = tf.math.top_k(preds, k=top_k)
    top_probs = top_probs.numpy().squeeze()
    top_indices = top_indices.numpy().squeeze()

    # +1 because label_map keys start at "1"
    classes = [str(i + 1) for i in top_indices]

    return top_probs, classes


def load_category_names(path):
    """
    Load a JSON file mapping label indices (as strings) to flower names.
    """
    with open(path, 'r') as f:
        class_names = json.load(f)
    return class_names


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict flower name from an image using a trained Keras model."
    )

    # Positional arguments
    parser.add_argument('image_path',
                        help='Path to input image.')
    parser.add_argument('model_path',
                        help='Path to saved Keras model (.h5).')

    # Optional arguments
    parser.add_argument('--top_k',
                        type=int,
                        default=5,
                        help='Return the top K most likely classes.')

    parser.add_argument('--category_names',
                        type=str,
                        default=None,
                        help='Path to JSON file mapping labels to flower names.')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Load model (with KerasLayer from TF Hub)
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

    # Load label map if provided
    class_names = None
    if args.category_names is not None:
        class_names = load_category_names(args.category_names)

    # Make prediction
    probs, classes = predict(args.image_path, model, top_k=args.top_k)

    # Print results
    print("\nTop {} predictions for image: {}\n".format(args.top_k, args.image_path))
    for prob, cls in zip(probs, classes):
        if class_names is not None and cls in class_names:
            label = class_names[cls]
        else:
            label = cls  # fallback to class index
        print(f"{label:30s} : {prob:.4f}")

    print()  # blank line at the end


if __name__ == '__main__':
    main()
