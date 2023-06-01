# Brain MRI Classification

This is a model that uses Tensorflow's Convolutional 2D system to classify MRI brain scans to detect whether or not there is a tumor present.

# Instructions

Due to GitHub's file size limit, the following will need to be done to ensure you have all the proper files to start classifying images

## Train the model

Edit lines 11, 13, 14, 16, and 17 in main.py to change the path to your desired install location and run the script

## Run the prediction script

Run predict.py and select an image placed in the dataset_uniform directory so that it can be ensured that the image being inputted is of size 200x200
# Dependencies
Pillow - Opening and changing the size of images
glob - Opening entire directories as lists
tensorflow - The ML Model
keras - using the Sequential model and various layer types
matplotlib - using pyplot to get a performance graph over time
numpy - formatting the data and presenting it to the model in a way that was comprehensible
os - various file management shenanigans
# Resources
https://www.tensorflow.org/tutorials/load_data/images + Various StackOverflow threads as issues came up
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection - Dataset
