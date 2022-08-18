import numpy as np
import tensorflow as tf
import skimage.measure

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_total = np.concatenate((x_train, x_test), axis=0)
# y_total = np.concatenate((y_train, y_test), axis=None)

digit_entropy = {}
for digit_index in range(10):
    digit_entropy[digit_index] = []

for x, y in zip(x_train, y_test):
    entropy = skimage.measure.shannon_entropy(x)
    digit_entropy[y].append(entropy)

for digit, entropy_list in digit_entropy.items():
    entropy_avg = sum(entropy_list) / len(entropy_list)
    print("Average entropy of " + str(digit) + ": " + str(entropy_avg))
