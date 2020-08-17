import argparse
import pathlib
import tensorflow as tf
from tensorflow import keras

# Arguments
parser = argparse.ArgumentParser(
    description='Trains a neural network model with a dataset and exports it')

parser.add_argument('--output', dest='file_name',
                    default="model",
                    help='name of the json and h5 files to which the neural network\'s model will be exported ('
                         'default: "model")')

parser.add_argument('--epochs', dest='epochs',
                    default="20",
                    help='number of training epochs (default: 20)')

parser.add_argument('--classes', dest='class_count',
                    default="1",
                    help='number of classes (default: 1)')

parser.add_argument('--model', dest='model_name',
                    default="MobileNetV2",
                    help='name of the neural network\'s model (default: "MobileNetV2")')

parser.add_argument('--data-dir', dest='dataset_directory',
                    default="dataset/",
                    help='directory of the dataset used for training (default: "dataset/")')

parser.add_argument('--yes', '-y', type=bool, default=False, help='skips the model confirmation if set to true')


# Creates a mobilenet v2 base model
def get_mobilenet_v2_model(img_shape):
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                   include_top=False, weights='imagenet')
    base_model.trainable = False
    return base_model


def get_inception_res_net_v2_model(img_shape):
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=img_shape,
                                                         include_top=False, weights='imagenet')
    base_model.trainable = False

    return base_model


def get_inception_v3_model(img_shape):
    base_model = tf.keras.applications.InceptionV3(input_shape=img_shape,
                                                   include_top=False, weights='imagenet')
    base_model.trainable = False

    return base_model


# Loads the dataset and returns a tuple of train data and validation data
def load_dataset(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training",
        seed=123, image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation",
        seed=123, image_size=(img_height, img_width),
        batch_size=batch_size)
    train_labels = train_ds.class_names
    val_labels = val_ds.class_names

    # Scale the RGB values to 0-1.0 range
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Cache and prefetch data for better training performance
    autotune_buffer_size = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=autotune_buffer_size)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune_buffer_size)

    return (train_labels, train_ds), (val_labels, val_ds)


# Creates the classifier and applies it to the base model
def apply_classifier(base_model, class_count):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(class_count)
    model = tf.keras.Sequential([
        base_model, global_average_layer,
        prediction_layer
    ])
    return model


# Main
def main():
    args = parser.parse_args()
    img_size = 75
    img_shape = (img_size, img_size, 3)
    BATCH_SIZE = 128
    class_count = int(args.class_count)
    epochs = int(args.epochs)

    # Load the data
    data_dir = pathlib.Path(args.dataset_directory)
    (train_labels, train_ds), (val_labels, val_ds) = load_dataset(data_dir, img_size, img_size, BATCH_SIZE)
    print("training labels(" + str(len(train_labels)) + "): ", train_labels)
    print("validation labels(" + str(len(val_labels)) + "): ", val_labels)

    # Load the base model and then apply the classifier
    model_switcher = {
        "MobileNetV2": get_mobilenet_v2_model(img_shape),
        "InceptionResNetV2": get_inception_res_net_v2_model(img_shape),
        "InceptionV3": get_inception_v3_model(img_shape)
    }
    model = model_switcher.get(args.model_name)
    model = apply_classifier(model, class_count)

    # Display the model and ask whether it is correct
    model.summary()
    if not args.yes:
        decision = input("Proceed with this model? (y/n):")
        if decision == '' or decision[0] != 'y':
            return

    # Train the model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_ds, epochs=epochs,
                        validation_data=val_ds)

    acc = history.history['accuracy']
    print("Training accuraccy: ", acc)
    print("Testing the accuracy with test dataset")

    print("Saving the model to " + args.file_name + ".json and its weights to " + args.file_name + ".h5...")

    model_json = model.to_json()
    with open(args.file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(args.file_name + ".h5")


if __name__ == "__main__":
    main()
