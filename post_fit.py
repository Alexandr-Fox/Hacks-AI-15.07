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

reconstructed_model = models.load_model("fit_4/best_model_0.08.h5")

# data = pd.read_csv("covid_data_train_4.csv")
data = pd.read_csv("covid_data_train_4.csv")
# data = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")

data.head()
data_features = data.copy()
data_labels = data_features.pop('inf_rate')
# data_labels = data_features.pop('survived')
X_train, X_test, y_train, y_test = train_test_split(
    data_features, data_labels, test_size=0.3, random_state=0)
# X_train = X_test = data_features
# y_train = y_test = data_labels
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
    name: np.array(value) for name, value in X_train.items()
}
data_features_dict_test = {
    name: np.array(value) for name, value in X_test.items()
}
features_dict = {
    name: values[:1] for name, values in data_features_dict.items()
}
data_preprocessing(features_dict)

callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.001,
        patience=2,
        min_lr=0.1
    ),
    ReduceLROnPlateau(
        monitor='loss',
        factor=0.001,
        patience=2,
        min_lr=0.1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=40,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    ),
    EarlyStopping(
        monitor="loss",
        patience=40,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    ),
    # EarlyStopping(
    #     monitor="val_mean_absolute_error",
    #     patience=5,
    #     restore_best_weights=True,
    #     min_delta=0.01,
    #     verbose=1
    # ),
    ModelCheckpoint(
        # filepath="best_model _4_{val_mean_absolute_error:.2f}.h5",
        filepath="fit_4/best_model_{val_mean_absolute_error:.2f}.h5",
        monitor="val_mean_absolute_error",
        mode="min",
        save_best_only=True
    ),
    TensorBoard(
        log_dir='./logs'
    )
]
reconstructed_model.summary()
history = reconstructed_model.fit(
    x=data_features_dict,  # Признаки
    y=y_train,  # Вектор целей
    epochs=300,  # Количество эпох
    # steps_per_epoch=20,
    verbose=1,  # Вывода нет
    batch_size=30,  # Количество наблюдений на пакет
    validation_data=(data_features_dict_test, y_test),  # Тестовые данные
    # use_multiprocessing=True, # паралелизм
    callbacks=callbacks  # отсечение
)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss, MAE')
plt.ylabel('Loss, MAE')
plt.xlabel('Epoch')
plt.legend(
    [
        'mean_absolute_error',
        'val_mean_absolute_error',
        'loss',
        'val_loss'
    ],
    loc='upper left'
)
plt.show()
with tf.device('/CPU:0'):
    predicted_target = reconstructed_model.predict(data_features_dict_test)
    c = 0
    c1 = 0
    c5 = 0
    c2 = 0
    y_test = y_test.tolist()
    y_train = y_train.tolist()
    summ = 0
    for i in range(len(predicted_target)):
        summ += abs(predicted_target[i][0] - y_test[i])
        print(
            f"{round(float(predicted_target[i][0]), 6)} - {round(y_test[i], 6)} - {round(abs(predicted_target[i][0] - y_test[i]), 6)}")
        if abs(predicted_target[i][0] - y_test[i]) < 0.001:
            c2 += 1
        if abs(predicted_target[i][0] - y_test[i]) < 0.01:
            c1 += 1
        if abs(predicted_target[i][0] - y_test[i]) < 0.1:
            c += 1
        if abs(predicted_target[i][0] - y_test[i]) < 0.15:
            c5 += 1
    print(summ / len(predicted_target))
    print(c, len(predicted_target))
    print(c1, len(predicted_target))
    print(c2, len(predicted_target))
    print(c5, len(predicted_target))
    predicted_target = reconstructed_model.predict(data_features_dict)
    c = 0
    c1 = 0
    c2 = 0
    c5 = 0
    summ = 0
    for i in range(len(predicted_target)):
        summ += abs(predicted_target[i][0] - y_train[i])
        if abs(predicted_target[i][0] - y_train[i]) < 0.001:
            c2 += 1
        if abs(predicted_target[i][0] - y_train[i]) < 0.01:
            c1 += 1
        if abs(predicted_target[i][0] - y_train[i]) < 0.1:
            c += 1
        if abs(predicted_target[i][0] - y_train[i]) < 0.05:
            c5 += 1
    print(summ / len(predicted_target))
    print(c, len(predicted_target))
    print(c1, len(predicted_target))
    print(c2, len(predicted_target))
    print(c5, len(predicted_target))
