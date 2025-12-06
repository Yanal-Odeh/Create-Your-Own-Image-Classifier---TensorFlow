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
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img.numpy()


def predict(image_path, model, top_k=5):
    """
    Predict the top K most probable classes for an image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    model : tf.keras.Model
        Trained Keras model.
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    probs : np.ndarray
        Top K probabilities.
    classes : list of str
        Corresponding class indices as strings (matching label_map.json keys).
    """
    # Load image
    im = Image.open(image_path)
    np_image = np.asarray(im)

    # Preprocess
    processed_image = process_image(np_image)

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    input_tensor = np.expand_dims(processed_image, axis=0)

    # Predictions
    preds = model.predict(input_tensor)  # shape: (1, num_classes)
    preds = np.squeeze(preds)           # shape: (num_classes,)

    # Ensure top_k is not larger than number of classes
    top_k = min(top_k, preds.shape[0])

    # Get top K
    top_probs, top_indices = tf.math.top_k(preds, k=top_k)
    top_probs = top_probs.numpy()
    top_indices = top_indices.numpy()

    # IMPORTANT: NO +1 OFFSET
    # label_map.json now starts at 0, so indices match directly
    classes = [str(i) for i in top_indices]

    return top_probs, classes


def load_category_names(json_path):
    """
    Load a JSON file mapping label indices (as strings) to flower names.
    """
    with open(json_path, 'r') as f:
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
    # Parse CLI args
    args = parse_args()

    # Load model (with TF Hub KerasLayer)
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )

    # Load label map if provided
    class_names = None
    if args.category_names is not None:
        class_names = load_category_names(args.category_names)

    # Predict
    probs, classes = predict(args.image_path, model, top_k=args.top_k)

    # Print results
    print("\nTop {} predictions for image: {}\n".format(args.top_k, args.image_path))

    for prob, cls in zip(probs, classes):
        if class_names is not None and cls in class_names:
            label = class_names[cls]   # NO +1 HERE
        else:
            label = cls               # fallback to raw index

        print(f"{label:30s} : {prob:.4f}")

    print()  # blank line


if __name__ == '__main__':
    main()
