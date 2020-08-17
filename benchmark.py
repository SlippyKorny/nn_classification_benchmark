import argparse
import os
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
from PIL import Image

# Arguments
parser = argparse.ArgumentParser(
    description='Checks if the parking spot was taken with the use of the passed model and weights file')

parser.add_argument('--multi-data-dir', dest='multi_data_directory', default="multi label classification data/",
                    help='path to the image with data for classification (default: "multi label classification data/")')

parser.add_argument('--multi-model-dir', dest='multi_model_directory', default='multi label classification models/',
                    help='path to the folder with all the multi label models (default: "multi label classification '
                         'models/")')

parser.add_argument('--binary-model-dir', dest='binary_model_directory', default='binary classification models/',
                    help='path to the folder with all the multi label models (default: "binary classification '
                         'models/")')

parser.add_argument('--binary-data-dir', dest='binary_data_directory', default="binary classification data/",
                    help='path to the image with data for classification (default: "binary classification data/")')

# Classes
occupation_labels = ['free', 'occupied']  # TODO: Check if this is correctly labeled

traffic_sign_labels = ['ampeln', 'doppelkurve_links', 'einschränkungsende', 'einseitig_(rechts)_verengte_fahrbahn', 'fußgänger',
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


# Model loading
def load_model(json_file_path, h5_file_path):
    # Loads and compiles the multi label classification model from the given files
    json_model = None
    with open(json_file_path, 'r') as json_file:
        json_model = json_file.read()

    model = model_from_json(json_model)
    model.load_weights(h5_file_path)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def get_model_data(model_dir_path):
    # Extracts all the models of a given type and their data in the given folder
    model_label = model_dir_path.split('/')[1]
    epoch_model_dict = {}
    json_dict = {}
    h5_dict = {}
    model_dir = os.listdir(model_dir_path)

    for item in model_dir:
        file_path = model_dir_path+'/'+item
        if os.path.isfile(file_path):
            epochs = item[0:2]
            if item.endswith('.json'):
                json_dict[epochs] = file_path
            elif item.endswith('.h5'):
                h5_dict[epochs] = file_path

    for key, val in json_dict.items():
        model = load_model(val, h5_dict[key])
        epoch_model_dict[key] = model
    return model_label, epoch_model_dict


def get_models(models_dir_path):
    # Returns the dictionary of all models and their data
    models_dict = []
    models_dir = os.listdir(models_dir_path)
    for item in models_dir:
        if os.path.isdir(models_dir_path+'/'+item):
            label, model_dict = get_model_data(models_dir_path+'/'+item)
            models_dict[label] = model_dict

    return models_dict


# Benchmarking
def load_images_with_classes(data_directory):
    # return [label, [Image]]
    image_dict = {}
    img_dataset_dir = os.listdir(data_directory)
    for item in img_dataset_dir:
        if os.path.isdir(item):
            class_dir = os.listdir(data_directory+item)
            key = item.replace('/', '', 1)
            image_dict[key] = []
            for file in class_dir:
                if os.path.isfile(file):
                    image = np.array(Image.open(data_directory + item + file)) / 255.0
                    image_dict[key].append(image)
    return image_dict


def benchmark(models, images):
    # Performs a benchmark of the speed of the neural networks with the passed models on the passed images
    pass

# Main
def main():
    args = parser.parse_args()
    multi_label_classification_models = get_models(args.multi_model_directory)
    binary_classification_models = get_models(args.binary_model_directory)


if __name__ == "__main__":
    main()
