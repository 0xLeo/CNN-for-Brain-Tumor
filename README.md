# CNN-for-Brain-Tumor
This is a simple Convolutional Neural Network Model for Brain Tumor Classification into five tumour types. 


# Requirements
1. Python 2.7.x or Python 3.5.x or larger
2. Tensorflow v1.0+ and Tensorflow GPU version to run on a graphic card
3. Scikit learn libraries
4. Numpy and Scipy
5. TFLearn (http://tflearn.org/installation/)

Installation of Tensorflow and other libraries can be found in their websites.

# Dataset
The dataset used for this research project is available under the link https://figshare.com/articles/brain_tumor_dataset/1512427/5  
It can be converted to images by using Matlab and running script `dataset_mat_to_img.m`.

# Execution
1. Make sure all the libaries are installed

2. Use create_dataset.py to convert your image dataset to a .pkl file. 

3. Run the proposed model 
```
 $ python proposed_model.py my_dataset.pkl
 ```
 
 # Altering the program
 You can use file proposed_model.py to change the parameters in it to test for yourown dataset. 
