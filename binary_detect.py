import argparse
import tensorflow as tf
from keras.models import model_from_json

import numpy as np
from PIL import Image

# Arguments
parser = argparse.ArgumentParser(
    description='Checks if the parking spot was taken with the use of the passed model and weights file')

parser.add_argument('-m', '--model', dest='model_path', default="model.json",
                    help='path to the model file (default: "model.json")')

parser.add_argument('-w', '--weights', dest='weights_path', default="model.h5",
                    help='path to the weights file (default: "model.h5")')

parser.add_argument('-i', '--image', dest='img_path', default="image.jpg",
                    help='path to the image that should be classified (default: "image.jpg")')


def get_model(model_path, weights_path):
    json_model = None
    with open(model_path, 'r') as json_file:
        json_model = json_file.read()

    model = model_from_json(json_model)
    model.load_weights(weights_path)
    print("Loaded from disk")

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print("compiled the model")
    return model


# Main
def main():
    args = parser.parse_args()
    model = get_model(args.model_path, args.weights_path)
    image = np.array(Image.open(args.img_path)) / 255.0
    print("min: ", np.min(image), "max:", np.max(image))

    prediction = model.predict(np.array([image]))
    predicted_class = None
    if prediction > 0:
        predicted_class = "Occupied"
    else:
        predicted_class = "Free"
    print("Prediction: ", predicted_class)


if __name__ == "__main__":
    main()
