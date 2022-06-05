from keras.layers import Rescaling, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation
from keras.applications import EfficientNetB0
from keras.models import Sequential
from flask import *
import numpy as np
import pandas as pd
from PIL import Image
import os
import librosa
from librosa.feature import melspectrogram
import warnings
from uuid import uuid4
import sklearn
import tensorflow as tf

most_represented_birds = ['American Crow', 'Andean Solitaire', 'Bananaquit',
                              'Black-crowned Night-Heron', 'Yellow-throated Toucan',
                              'Bright-rumped Attila', 'Buff-rumped Warbler', 'Canada Goose',
                              'Chestnut-backed Antbird', 'Chestnut-crowned Antpitta',
                              'Dark-eyed Junco', 'Dusky-capped Flycatcher',
                              'Eurasian Collared-Dove', 'Ferruginous Pygmy-Owl',
                              'Green-winged Teal', 'Grayish Saltator', 'Great Thrush',
                              'Gray Catbird', 'Greater White-fronted Goose', 'House Finch',
                              'Laughing Falcon', 'Long-billed Gnatwren', 'Marsh Wren',
                              'Mute Swan', 'Northern Flicker', 'Russet-backed Oropendola',
                              'Savannah Sparrow', 'Scale-crested Pygmy-Tyrant',
                              'Slate-throated Redstart', 'Smooth-billed Ani',
                              'Southern Beardless-Tyrannulet', 'Southern Lapwing',
                              'Striped Cuckoo', "Swainson's Thrush", 'Tropical Kingbird',
                              'Western Meadowlark', 'White-crowned Sparrow']


def get_sample(filename, bird, output_folder):
    wave_data, wave_rate = librosa.load(filename)
    wave_data, _ = librosa.effects.trim(wave_data)
    sample_length = 5 * wave_rate
    samples_from_file = []
    N_mels = 216
    for idx in range(0, len(wave_data), sample_length):
        song_sample = wave_data[idx:idx + sample_length]
        if len(song_sample) >= sample_length:
            mel = melspectrogram(song_sample, n_mels=N_mels)
            db = librosa.power_to_db(mel)
            normalised_db = sklearn.preprocessing.minmax_scale(db)
            filename = str(uuid4()) + ".jpg"
            db_array = (np.asarray(normalised_db) * 255).astype(np.uint8)
            db_image = Image.fromarray(np.array([db_array, db_array, db_array]).T)
            db_image.save("{}{}".format(output_folder, filename))
            samples_from_file.append(
                {"song_sample": "{}{}".format(output_folder, filename), "db": db_array, "bird": bird})
    return samples_from_file

def load_model():
    input_shape = (216, 216, 3)
    effnet_layers = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)

    for layer in effnet_layers.layers:
        layer.trainable = True

    dropout_dense_layer = 0.3

    model = Sequential()
    model.add(Rescaling(1. / 255, input_shape=(216, 216, 3)))
    model.add(effnet_layers)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_dense_layer))
    model.add(Dense(37, activation=(tf.nn.softmax)))
    model.summary()

    model.load_weights(r"venv/static/model.h5")
    return model

def predict(file):
    warnings.filterwarnings("ignore")
    output_folder = r"venv/image/"
    output_folder += str(uuid4())
    output_folder += '/'
    os.mkdir(output_folder)
    get_sample(file, 'unknown', output_folder)
    test_img = os.listdir(output_folder)

    result = []
    num_img = 0
    for img in test_img:
        img_path = output_folder + '/' + img
        img = tf.keras.utils.load_img(
            img_path, target_size=(216, 216)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        result.append(score)
        num_img += 1

    result = pd.DataFrame(result)
    result = pd.DataFrame(result.T)
    result = result.sum(axis=1) / num_img
    return result


app = Flask(__name__)
audiotype = ['audio/midi ','audio/mpeg','audio/ogg','audio/m4a ','audio/x-flac','audio/x-wav',]

@app.route('/',methods = ['GET','POST'])
def result():
    string = "Please upload an audio file."
    if request.method == 'POST':
        uploaded_file = request.files['file']
        filetype = uploaded_file.mimetype
        if uploaded_file.filename != '':
            if filetype not in audiotype:
                string = "This is not an audio file, please upload again."
            else:
                result = predict(uploaded_file)
                prediction = most_represented_birds[np.argmax(result)]
                string = "This audio most likely belongs to {}.".format(prediction)
        else:
            string = "Unknown error, please try again."
    return render_template('page.html', string = string)

if __name__ == '__main__':
    model = load_model()
    app.run(port= 8080, debug=True)


