# MNIST-Exploration
This repo is an exploration of idea of specialization with neural networks trained on the MNIST dataset.

Dataset info
    MNIST contains images of handwritten digits from 500 writers
    The dataset includes 60,000 training and 10,000 test images
    Input data
        Each image has a size of 28x28
        Each image is greyscale with values ranging from [0, 255] of type uint8
        0 denotes an empty pixel (white), 255 is a fully colored in pixel (black)
    Ouput data
        There are 10 labels, one for each digit
        The labels are type uint8

Parameter of dense layer: (current layer neurons * previous layer neurons) + current layer neurons
Parameter added by additional neuron in middle layer: previous layer neurons + previous layer neurons + 1

TODO:
    Add InputShape to settings?
    Add timer to logger to measure train and predict time?