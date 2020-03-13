from __future__ import print_function
import cv2
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_test_data(size):
  test_data_dir = os.path.join(os.getcwd(), 'test')
  imgs = [cv2.imread(os.path.join(test_data_dir, str(i) + '.png'), 0).reshape(28, 28, 1) for i in range(size)]
  return np.array(imgs, dtype='float32')

def ensemble_vote(ensemble_predictions):
  model_acc = [0.9449, 0.9443, 0.9443, 0.9434, 0.9433]
  alpha = [0.5 * math.log(acc / (1 - acc)) for acc in model_acc]
  predictions = []
  for image_predictions in np.transpose(ensemble_predictions):
    class_score = [0]*10
    for model, pred in enumerate(image_predictions):
      class_score[pred] += alpha[model]
    predictions.append(np.argmax(class_score))
  return predictions

if __name__ == '__main__':
  model_name = 'fashion_mnist_cnn'  # best model 0 (94.49) > 1 (94.43) > 2 (94.43) > 3 (94.34) > 4 (94.33)
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  ensemble_num = 3
  # model_path = os.path.join(save_dir, model_name + '.h5')

  print('loading test data...')
  test_data = load_test_data(10000)
  print('done')

  print('\nloading {} model parameters...'.format(model_name))
  # model = load_model(model_path)
  models = [load_model(os.path.join(save_dir, model_name + str(i) + '.h5')) for i in range(ensemble_num)]
  print('done')

  print('\nmaking inference:')
  ensemble_predictions = [np.argmax(cnn.predict(test_data, batch_size=128, verbose=1), axis=1) for cnn in models]
  predictions = ensemble_vote(ensemble_predictions)

  with open('prediction.txt', 'w') as f:
    for pred in predictions:
        f.write('{}\n'.format(pred))

  print('\nSaved predictions to prediction.txt')
