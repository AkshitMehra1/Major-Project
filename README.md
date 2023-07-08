# Hyperparameter Optimization in Deep Learning using Genetic Algorithm (GA)

This project focuses on optimizing hyperparameters in deep learning models using a Genetic Algorithm (GA). The goal is to find the optimal hyperparameter configuration for a Convolutional Neural Network (CNN) in the context of Alzheimer's disease detection. The project explores the effectiveness of the genetic algorithm in automatically searching the hyperparameter space and improving the performance of the CNN model.

## Abstract

This project focuses on Hyperparameter Optimization in Deep Learning using a Genetic Algorithm (GA). In deep learning, hyperparameters play a crucial role in determining the performance of models, and finding the optimal values can be a challenging task. The Genetic Algorithm is a powerful optimization technique inspired by natural selection and evolution. This project specifically applies the genetic algorithm to optimize hyperparameters for a Convolutional Neural Network (CNN) in the context of Alzheimer's disease detection. Alzheimer's disease is a neurodegenerative disorder that affects millions of people worldwide. The goal is to classify brain images into one of four classes: 'Mild_Demented', 'Moderate_Demented', 'Non_Demented', and 'Very_Mild_Demented'. The hyperparameters considered in this study include the number of filters, kernel size, dropout rate, number of dense units, learning rate, and batch size. By exploring different combinations of these hyperparameters, the genetic algorithm aims to discover the configuration that yields the highest accuracy on a validation set. The best individual from the final generation is then used to train the CNN model. The project demonstrates the effectiveness of the genetic algorithm in automatically searching the hyperparameter space and finding optimal configurations for Alzheimer's disease classification. The obtained results showcase the significance of hyperparameter optimization and the utility of genetic algorithms in enhancing the performance of deep learning models. The genetic algorithm achieved a high testing accuracy of 0.9922, outperforming Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO) in terms of achieving the best results. Throughout the project, training and validation accuracy and loss graphs are analyzed to gain insights into the model's performance over epochs. This analysis allows for comprehensive evaluation and comparison of different hyperparameter optimization techniques.

## Project Structure

The project contains a Jupyter Notebook file, `code.ipynb`, which includes code and outputs related to hyperparameter optimization using Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO). The notebook file provides step-by-step explanations of the optimization algorithms and their implementation.

## Dataset Preparation

1. Prepare a dataset folder containing brain MRI images. The images should be organized into four folders representing the four classes: 'Mild_Demented', 'Moderate_Demented', 'Non_Demented', and 'Very_Mild_Demented'.
2. Ensure that the images are resized to 128 by 128 pixels.

## Usage

To run the code and reproduce the results, follow these steps:

1. Upload the `code.ipynb` notebook to Google Colab.
2. Upload the dataset folder to your Google Drive. Make a note of the directory name or path where the dataset is located.
3. Set the directory path to the dataset folder in the code by updating the appropriate variable or path information in the notebook.
4. Execute the cells in the notebook sequentially, starting from the top, to run the genetic algorithm, particle swarm optimization, and ant colony optimization for hyperparameter optimization, as well as train the CNN models.
5. Analyze the outputs, including training and validation accuracy and loss graphs, to evaluate the performance of the optimized models.

## Results

The project demonstrates the effectiveness of the genetic algorithm in automatically searching the hyperparameter space and finding optimal configurations for Alzheimer's disease classification. It achieves a high testing accuracy of 0.9722, outperforming other optimization techniques such as Particle Swarm Optimization (PSO) and Ant Colony Optimization (ACO).
