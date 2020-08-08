import argparse
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers

# keras = tf.keras

# Arguments
parser = argparse.ArgumentParser(description='Trains a convolutional neural network model with the preprocessed CNR-EXT dataset and exports it')

parser.add_argument('--output', dest = 'file_name', 
                    default = "model", 
                    help = 'name of the json and h5 files to which the cnn model will be exported (default: "model")')

parser.add_argument('--model', dest = 'model_name', 
                    default = "MobileNetV2", 
                    help = 'name of the cnn model (default: "MobileNetV2")')

parser.add_argument('--data-dir', dest = 'dataset_directory', 
                    default = "dataset/", 
                    help = 'directory of the dataset used for training (default: "dataset/")')

# Creates a mobilenet v2 base model
def get_mobilenet_v2_model(IMG_SIZE):
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE,
        include_top = False, weights = 'imagenet')
    base_model.trainable = False
    return base_model

def foo():
    return 

# Loads the dataset and returns a tuple of train data and validation data
def load_dataset(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training",
        seed=123, image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, validation_split = 0.2, subset = "validation",
        seed = 123, image_size = (img_height, img_width),
        batch_size = batch_size)

    # Scale the RGB values to 0-1.0 range
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Cache and prefetch data for better training performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    return (train_ds, val_ds)

# Creates the classifier and applies it to the base model
def apply_classifier(base_model):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(1)
    model = tf.keras.Sequential([
        base_model, global_average_layer,
        prediction_layer
    ])
    return model

# Main
def main():
    args = parser.parse_args()
    IMG_SHAPE = 75
    BATCH_SIZE = 128

    # Load the data
    data_dir = pathlib.Path(args.dataset_directory)
    train_ds, val_ds = load_dataset(data_dir, IMG_SHAPE, IMG_SHAPE, BATCH_SIZE)

    # Load the base model and then apply the classifier
    model_switcher = {
        "MobileNetV2": get_mobilenet_v2_model(IMG_SHAPE)#,
        # "foo": foo()
    }
    model = model_switcher.get(args.model_name)
    model = apply_classifier(model)

    # Display the model and ask whether it is correct
    model.summary()
    decision = input("Proceed with this model? (y/n):")
    if decision == '' or decision[0] != 'y':
        return

    # Train the model
    initial_epochs = 40
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(train_ds, epochs=initial_epochs,
                    validation_data=val_ds)
    
    acc = history.history['accuracy']
    print("Training accuraccy: ", acc)
    print("Testing the accuracy with test dataset")
    

    print("Saving the model to " +  args.file_name + ".json and its weights to " + args.file_name + ".h5...")
    
    model_json = model.to_json()
    with open(args.file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(args.file_name + ".h5")


if __name__ == "__main__":
    main()
