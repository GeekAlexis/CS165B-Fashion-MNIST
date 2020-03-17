# Fashion-MNIST-CNN
TensorFlow TPU model for Fashion-MNIST
## CNN model
- 4-Conv and 3-FC with Relu activation, dropout and batch normalization
- 1/5 of the training data extracted as validation data to reduce overfitting
- Horizontal flip, shift, rotate and zoom augmentation used to balance training data and improve accuracy
- Accuracy on the test set: 94.8%
