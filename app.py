import os

import pandas
from flask import Flask, flash, request, redirect, url_for, render_template
# объясняется ниже
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers.experimental.preprocessing
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.callbacks_v1 import TensorBoard
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics, regularizers, Sequential, initializers
from tensorflow.keras.layers import Concatenate, Dense, Dropout
from tensorflow.keras import losses, Model, optimizers, models
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import plot_model

UPLOAD_FOLDER = '/home/alexandr-fox/PycharmProjects/flaskProject1/'
# расширения файлов, которые разрешено загружать
ALLOWED_EXTENSIONS = {'csv'}
app = Flask(__name__)
# конфигурируем
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict(path):
    reconstructed_model = models.load_model("best_model_0.06.h5")
    # plot_model(model=reconstructed_model, rankdir="LR", dpi=72, show_shapes=False)
    # cols = pandas.read_csv(path, nrows=0).columns
    # print(cols)
    # cols = cols[cols.str.contains("")].tolist()
    data = pd.read_csv(path)

    data.head()
    data_features = data.copy()
    inputs = {}

    for name, column in data_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    numeric_inputs = {name: input for name, input in inputs.items()
                      if input.dtype == tf.float32}

    x = Concatenate()(list(numeric_inputs.values()))
    norm = preprocessing.Normalization()
    norm.adapt(np.array(data[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)
    preprocessed_inputs = [all_numeric_inputs]
    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue
        lookup = tensorflow.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=np.unique(data_features[name]))
        one_hot = tensorflow.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)
    preprocessed_inputs_cat = Concatenate()(preprocessed_inputs)

    data_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

    data_features_dict = {
        name: np.array(value) for name, value in data_features.items()
    }
    features_dict = {
        name: values[:1] for name, values in data_features_dict.items()
    }
    data_preprocessing(features_dict)

    res = []
    # coordinates = []
    predicted_target = reconstructed_model.predict(data_features_dict)
    for i in range(len(predicted_target)):
        res.append([[data_features_dict['lng'][i], data_features_dict['lat'][i]], round(predicted_target[i][0], 4),
                    data_features_dict['name'][i].replace('&#39;', '')])
        # coordinates.append()
    print(res)
    return res


def allowed_file(filename):
    """ Функция проверки расширения файла """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/maps', methods=['GET', 'POST'])
def maps():
    if request.method == 'POST':
        # проверим, передается ли в запросе файл
        if 'file' not in request.files:
            # После перенаправления на страницу загрузки
            # покажем сообщение пользователю
            flash('Не могу прочитать файл')
            return redirect(request.url)
        file = request.files['file']
        # Если файл не выбран, то браузер может
        # отправить пустой файл без имени.
        if file.filename == '':
            flash('Нет выбранного файла')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # безопасно извлекаем оригинальное имя файла
            filename = secure_filename(file.filename)
            # сохраняем файл
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            try:
                res = predict(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except:
                return render_template('index.html', error="Неверный формат датасета")
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # если все прошло успешно, то перенаправляем
            # на функцию-представление `download_file`
            # для скачивания файла
            return render_template('maps.html', coords=res)
    res = predict(os.path.join(app.config['UPLOAD_FOLDER'], "covid_data_test_1.csv"))
    return render_template('maps.html', coords=res)


@app.route('/', methods=['GET'])
def hello_world():  # put application's code here
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
