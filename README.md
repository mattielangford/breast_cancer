# Breast Cancer Deep Neural Network 
This project was completed in honor of Women's International Day. 

#### -- Project Status: Completed

## Project Intro/Objective
The purpose of this project is build a neural network to classify malignant/benign tumors following a positive BRCA gene test. 

### Methods Used
* Data Scaling
* Data Cleaning
* Date Visualization
* Machine Learning
* Data Visualization
* Predictive Modeling

### Technologies
* Python
* PyCharm 

## Project Description
This project was built using Sklearn's dataset for tumors removed from women who have tested positive for the BRCA gene.  The data set includes 30+ columns regarding measurements about the tumor, as well as a classified outcome (malignant: 1; benign: 0). A neural network with 30 input features, 3 hidden layers, and 1 output was able to predict with 92% accuracy the outcome of unlabeled data.  This accuracy score could be further optimized by performing a grid_search on the three different neural_network functions include.

## Needs of this project

- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- basic knowledge of breast cancer and BRCA gene
- advanced derivatives and optimization functions 

## Getting Started

1. Clone this repo 
2. Raw Data can be imported from sklearn.datasets.load_breast_cancer

## Required Python Dependencies:
1. pandas
2. numpy
3. scipy
4. sklearn
5. seaborn
6. matplotlib


## Mathematical Conclusions:
* Original activation function Sigmoid was replaced with tanh due to steeper gradient descent of derivative and faster learning rate. 
* Data was scaled and weights were initialized to random numbers within the range [0:1]
* A squared-error calculation was used to discover the cost of prediction vs. target. 
* Back propagation step was used to optimize cost by taking the derivative of each weight/bias with respect to cost. 
* A learning rate of 0.005 seemed to be optimal for accuracy_score. 
