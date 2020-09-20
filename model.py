import tensorflow as tf
from tensorflow.keras.losses import mse

import numpy as np
import time
from IPython import display

from config import Configuration as config

from optimizer import CG


# model definition
class Model(tf.keras.Model):
  def __init__(self, params):
    super(Model, self).__init__()
    self.inference_net = tf.keras.Sequential(
      [
        tf.keras.layers.Dense(
          units=params[0], #['units_1'],
          input_dim=12,
          activation=params[1] #['activation_1']
          ),
        tf.keras.layers.Dropout(
          rate=params[2] #['dropout_rate_1']
          ),
        tf.keras.layers.Dense(
          units=params[3], #['units_2'],
          activation=params[4] #['activation_2']
          ),
        tf.keras.layers.Dropout(
          rate=params[5] #['dropout_rate_2']
          ),
        tf.keras.layers.Dense(
          units=params[6], #['units_3'],
          activation=params[7] #['activation_3']
          ),
        tf.keras.layers.Dropout(
          rate=params[8] #['dropout_rate_3']
          ),
        tf.keras.layers.Dense(
          units=2,
          activation='linear')
      ]
    )
  
  def call(self, x):
    return self.predict(x)

  def predict(self, x):
    return self.inference_net(x)


def compute_loss(model, x, y):
    y_pred = model.call(x)
    return mse(y_pred, y)


@tf.function
def compute_apply_gradients(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train(model, data, epochs, params):
    # choose curves and target
    #data = data_pig_cala
    #labels = labels_5pc
    data, labels = data

    # split into train and test ---------(1)
    #idx = random.sample(range(len(data)), len(data))
    #train_idx, test_idx = idx[:2000], idx[2000:]

    # ShuffleSplit  ---------(2)
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=config.SEED)
    train_idx = []
    test_idx = []
    for train_idx, test_idx in ss.split(range(len(data))):
      train_idx, test_idx = train_idx, test_idx

    # data generation functions
    def gen_datum(idx, test=False):
      return data[idx], labels[idx]

    def gen_train_data():
      for idx in train_idx:
        yield gen_datum(idx, test=False)

    def gen_test_data():
      for idx in test_idx:
        yield gen_datum(idx, test=False)

    mse = tf.keras.losses.MeanSquaredError()
    # optimizer = tf.keras.optimizers.Adam(params[9]) #['learning_rate'])
    # optimizer = tf.keras.optimizers.SGD(params[9])
    # optimizer = AMSGrad(params[9])
    optimizer = CG(params[9])

    # keep track of losses
    loss_list=[]

    # training
    for epoch in range(1, epochs + 1):
      start_time = time.time()
      for train_x, train_y in gen_train_data():
        compute_apply_gradients(model, train_x, train_y, optimizer)
      end_time = time.time()

      if epoch % 1 == 0:
        loss = tf.keras.metrics.Mean()
        for test_x, test_y in gen_test_data():
          loss(compute_loss(model, test_x, test_y))
        loss_list.append(loss.result())
        display.clear_output(wait=True)
        print('Epoch: {}, Test loss: {}, '
          'time elapse for current epoch {}'
          .format(epoch, loss.result(), end_time - start_time))