# Best-Buddies

The main algorithm to find the neural best buddies between 2 pictures is in code.py

You can specify some useful parameters in the file options.json:
- gamma is the treshold for the relative activation under which we don't take into account the best buddies for the next layer
- layers are the selected layers for the computation of best buddies
- patch radius determine the size of the neighborhood around the best buddies where we will search the next ones in the following layer
- cluster is the number of points we want to keep at the end of the algorithm

# Selected Best Buddies

The code for selected best buddies, along with the data and the results used, are in the branch "topdown".

# Fine-tuning

We used the Deep Fashion database to fine tune VGG19 in the file fine_tuning.py

To use this script, you need to download:
- the folder img here http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- the file list_category_img.txt here http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

Then, you can specify how much data you want to use in the script parse.py, and run it to prepare the data.
Finally, you can use the script fine_tuning.py to train the last layers of VGG19 to classify the clothes from Deep Fashion. This will save the network, which can be used test_fine.py to find best buddies between two images

# Visualization_conv_cat
The code to visualize the outputs after each convolutional operation with the VGG19 CNN

# Visualization_filters
The code to visualize the filters of each layer of the VGG19 CNN


