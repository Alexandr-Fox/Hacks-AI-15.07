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

# reconstructed_model = models.load_model("fit/best_model_0.08.h5")
reconstructed_model = models.load_model("fit_4/best_model_0.06.h5")
data = pd.read_csv("covid_data_test_1.csv")
# data = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")

data.head()
data_features = data.copy()
# data_labels = data_features.pop('inf_rate')
# data_labels = data_features.pop('survived')
# X_train, data_features, y_train, data_labels = train_test_split(
#     data_features, data_labels, test_size=0.33, random_state=0)
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

print(data_features_dict)
with tf.device('/CPU:0'):
    predicted_target = reconstructed_model.predict(data_features_dict)
    c = 0
    c1 = 0
    c2 = 0
    c5 = 0
    summ = 0

    # data_labels = data_labels.tolist()
    for i in range(len(predicted_target)):
        print(predicted_target[i][0])

    print(len(predicted_target))
    #     summ += abs(predicted_target[i][0] - data_labels[i])
    #     if abs(predicted_target[i][0] - data_labels[i]) < 0.001:
    #         c2 += 1
    #     if abs(predicted_target[i][0] - data_labels[i]) < 0.01:
    #         c1 += 1
    #     if abs(predicted_target[i][0] - data_labels[i]) < 0.1:
    #         c += 1
    #     if abs(predicted_target[i][0] - data_labels[i]) < 0.15:
    #         c5 += 1
    # print(summ / len(predicted_target))
    # print(c, len(predicted_target))
    # print(c1, len(predicted_target))
    # print(c2, len(predicted_target))
    # print(c5, len(predicted_target))
