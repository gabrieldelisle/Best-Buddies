# -*- coding: utf-8 -*-
"""

@author: Aniss
"""

# Code adapted from https://github.com/himanshurawlani/convnet-interpretability-keras/blob/master/Visualizing%20filters/visualizing_convnet_filters.ipynb

import numpy as np
import keras
import matplotlib.pyplot as plt

from keras.applications import vgg19

# defining the VGG19 model in Keras

base_model = vgg19.VGG19(weights='imagenet', include_top=False)

np.save('base_vgg19.npy', base_model.get_weights())
base_model.save('base_vgg19.h5')


# loading the saved model

from keras.models import load_model

base_model = load_model('base_vgg19.h5')
base_model.set_weights(np.load('base_vgg19.npy')) 


# visualizing the model

from IPython.display import display, HTML
from keras.utils import plot_model

import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pyd

def plot_model_architecture(base_model, model_name):
    plot_model(base_model, show_shapes=True, to_file=model_name)
    display(HTML('<img src="{}" style="display:inline;margin:1px"/>'.format(model_name)))

#plot_model_architecture(base_model, 'base_vgg19_model.svg')
    
 
    
    


# dimensions of the generated pictures for each filter.
img_width = 128
img_height = 128

# this is the placeholder for the input images
input_img = base_model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in base_model.layers[1:]])




# maximizing the activation of specific filter

from keras import backend as K

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())





def gradient_ascent(iterate):
    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

#         print('------>Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break
        
    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
        
        
        



def build_nth_filter_loss(filter_index, layer_name):
    """
    We build a loss function that maximizes the activation
    of the nth filter of the layer considered
    """
    
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])
    
    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    return iterate        






# iterating over some number of filters in a given layer_name
    
layers = ['block1_conv1', 'block1_conv2',
          'block2_conv1', 'block2_conv2',
          'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4',
          'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4',
          'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4',]






import time

kept_filters = []
filters_dict = dict()
for layer_name in layers: # Beware, this loop takes like 4.5 hours with a good laptop...
    layer = base_model.get_layer(layer_name)
    print('Processing filter for layer:', layer_name)
    for filter_index in range(min(layer.output.shape[-1], 100)):
        # print('Processing filter %d' % filter_index)

        start_time = time.time()
        gradient_ascent(build_nth_filter_loss(filter_index, layer_name))
        end_time = time.time()

#         print('--->Filter %d processed in %ds' % (filter_index, end_time - start_time))
    filters_dict[layer.name] = kept_filters
    kept_filters = []



    
for layer_name, kept_filters in filters_dict.items():
    print(layer_name, len(kept_filters))
    


''' saving the objects

f1 = open('f1_filters_dict.pckl', 'wb')
pickle.dump(filters_dict, f1)
f1.close()

f2 = open('f2_kept_filters.pckl', 'wb')
pickle.dump(kept_filters, f2)
f2.close()


f3 = open('f3_layer_dict.pckl', 'wb')
pickle.dump(layer_dict, f3)
f3.close()

'''


# Stiching best filters on a black picture


from keras.preprocessing.image import save_img



def stich_filters(kept_filters, layer_name):
    # By default, we will stich the best 64 (n*n) filters on a 8 x 8 grid.
    n = int(np.sqrt(len(kept_filters)))
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img

    # save the result to disk
    save_img('stitched_filters_{}.pdf'.format(layer_name), stitched_filters)



    
for layer_name, kept_filters in filters_dict.items():
    print('Stiching filters for {}'.format(layer_name))
    stich_filters(kept_filters, layer_name)
    print('Completed.')




'''
# Visualizing the filters of each layer of VGG19 network

from keras.preprocessing import image

%matplotlib inline

filter_name = 'block5_conv4'

img = image.img_to_array(image.load_img('stitched_filters_{}.png'.format(filter_name))) /255.
plt.figure(figsize=(17,17))
plt.imshow(img)
plt.title(filter_name)
plt.grid(False)

'''