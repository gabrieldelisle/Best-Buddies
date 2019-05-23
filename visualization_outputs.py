# -*- coding: utf-8 -*-
"""

@author: Aniss
"""

# Code adapted from https://github.com/himanshurawlani/convnet-interpretability-keras/blob/master/Visualizing%20intermediate%20activations/visualizing_intermediate_activations.ipynb

import keras
keras.__version__

from keras.applications.vgg19 import VGG19

model = VGG19(weights='imagenet')
model.summary()

img_path = 'original_B.png'

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 224, 224, 3) using print(img_tensor.shape)


from keras import models

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[1:]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


# This will return a list of 25 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)


first_layer_activation = activations[0]
#print(first_layer_activation.shape)


import matplotlib.pyplot as plt

#visualizing the 3rd channel:
#plt.matshow(first_layer_activation[0, :, :, 3], cmap='plasma')
#plt.show()




# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[1:]:
    layer_names.append(layer.name)

#images_per_row = 16
images_per_row = 3

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
#    n_cols = n_features // images_per_row
    n_cols = images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 2.4 / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='plasma')
    plt.savefig(layer_name+".png")
