import argparse
import tensorflow as tf
from keras.models import model_from_json
import numpy as np
from PIL import Image
import sys

# Arguments
parser = argparse.ArgumentParser(
    description='Checks if the parking spot was taken with the use of the passed model and weights file')

parser.add_argument('-m', '--model', dest='model_path', default="model.json",
                    help='path to the model file (default: "model.json")')

parser.add_argument('-w', '--weights', dest='weights_path', default="model.h5",
                    help='path to the weights file (default: "model.h5")')

parser.add_argument('-i', '--image', dest='img_path', default="image.jpg",
                    help='path to the image that should be classified (default: "image.jpg")')

# Classes
classes = ['ampeln', 'doppelkurve_links', 'einschränkungsende', 'einseitig_(rechts)_verengte_fahrbahn', 'fußgänger',
           'gefahrstelle', 'kinder', 'kreisverkehr', 'kurve_links', 'kurve_rechts', 'radverkehr',
           'schleuder-_oder_rutschgefahr', 'schnee-_oder_eisglätte', 'stop', 'straßenarbeiten', 'unebene_fahrbahn',
           'verbot_der_einfahrt', 'verbot_für_lastkraftwagen', 'verboten', 'vorfahrt', 'vorfahrt_geben',
           'vorfahrtstraße', 'vorgeschriebene_fahrtrichtung_geradeaus',
           'vorgeschriebene_fahrtrichtung_geradeaus_oder_links', 'vorgeschriebene_fahrtrichtung_geradeaus_oder_rechts',
           'vorgeschriebene_fahrtrichtung_links', 'vorgeschriebene_fahrtrichtung_links_vorbei',
           'vorgeschriebene_fahrtrichtung_rechts', 'vorgeschriebene_fahrtrichtung_rechts_vorbei', 'wildwechsel',
           'zulässige_höchstgeschwindigkeit_100', 'zulässige_höchstgeschwindigkeit_120',
           'zulässige_höchstgeschwindigkeit_20', 'zulässige_höchstgeschwindigkeit_30',
           'zulässige_höchstgeschwindigkeit_50', 'zulässige_höchstgeschwindigkeit_60',
           'zulässige_höchstgeschwindigkeit_70', 'zulässige_höchstgeschwindigkeit_80',
           'zulässige_höchstgeschwindigkeit_80_ende', 'überholen_verboten_ende', 'überholen_verboten_ende_tir',
           'überholverbot', 'überholverbot_tir']


# Program
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

    predictions = model.predict(np.array([image]))[0]
    highest_val = -sys.maxsize - 1
    index_of_highest = -1

    for i in range(0, len(classes)):
        if predictions[i] > highest_val:
            highest_val = predictions[i]
            index_of_highest = i

    print("predicted: '", classes[index_of_highest], "' with the possibility of: ", highest_val)


if __name__ == "__main__":
    main()
