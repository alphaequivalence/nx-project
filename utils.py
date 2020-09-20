import pandas as pd
import numpy as np
from IPython import display

# load data
curves = [
  'ATGPig',
  'ATG5pc',
  'ATG10pc',
  'ATG15pc',
  'ATG20pc',
  'ATG25pc',
  'ATG35pc',
  'ATGcala'
]
columns = [
  'time',
  'temperature',
  'weight',
  'heat flow',
  'temperature difference (°C)',
  'temperature difference (µV)',
  'Sample purge flow',
  ''
]
TRAINING_COLUMNS = [
  'weight',
  'heat flow',
  'temperature difference (°C)',
  'Sample purge flow',
]
LABELLING_COLUMNS = [
  'weight',
  'temperature difference (°C)'
]


def load_dataset():
  dataset = dict.fromkeys(curves)
  for curve in curves:
    display.clear_output(wait=True)
    print('loading ', curve, '...')
    dataset[curve] = pd.read_csv('./data/'+curve+'.csv', delimiter=',', header=0, names=columns[:-1])
  return dataset


def select_curves(dataset, CURVES_OF_INTEREST = ['ATGPig', 'ATGcala'], TARGET_CURVE = 'ATG35pc'):
  temperature = np.expand_dims(dataset[TARGET_CURVE]['temperature'].to_numpy()[:3000], axis=1)
  column_35pc = 35 * np.expand_dims(np.ones(3000, dtype='float'), axis=1)
  column_0pc = np.expand_dims(np.zeros(3000, dtype='float'), axis=1)
  column_100pc = 100 * np.expand_dims(np.ones(3000, dtype='float'), axis=1)
  toto = np.concatenate(
    [
      temperature, column_35pc, column_0pc, column_100pc
    ],
    axis=1
  )

  data_pig_cala = np.concatenate(
    [
      dataset[curve][TRAINING_COLUMNS].to_numpy()[:3000]
      for curve in CURVES_OF_INTEREST
    ],
    axis=1
  )
  data_pig_cala = np.concatenate((toto, data_pig_cala), axis=1)
  data_pig_cala = np.expand_dims(data_pig_cala, axis=1)

  labels_35pc = dataset[TARGET_CURVE][LABELLING_COLUMNS].to_numpy()[:3000]

  return data_pig_cala, labels_35pc