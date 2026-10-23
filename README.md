# Muscle Activity Classification with a Multilayer Perceptron

This project was developed for a machine learning exam. The goal was to build and evaluate a **Multilayer Perceptron (MLP)** that classifies upper-body muscle activity from numerical input vectors.

The notebook documents the full process: exploring the data, preparing it for training, building a neural network, evaluating its predictions, and experimenting with hyperparameter tuning.

## The dataset

The dataset has **2,700 samples**, each represented by **132 numerical features**. The model predicts one of seven labels:

- Left bicep, shoulder, or tricep
- Right bicep, shoulder, or tricep
- Rest

The input data is stored in a [separate GitHub repository](https://github.com/barbara-cacace/vector_input_for_MLP_model_exam). The notebook downloads it automatically; you do not need to upload the dataset again.

## What I did

1. **Prepared the data:** inspected the inputs and labels, checked for missing values, encoded the classes, shuffled the samples, and split them into training, validation, and test sets. I applied MinMax scaling using the training set.
2. **Built an MLP:** used TensorFlow/Keras to create a feedforward neural network with dense layers and a seven-class softmax output. I also used dropout, batch normalization, and L2 regularization.
3. **Evaluated the model:** examined training and validation curves, test accuracy, F1 scores, a classification report, and a confusion matrix.
4. **Tuned the model:** used Optuna to test different numbers of layers and neurons, dropout rates, regularization strengths, learning rates, and batch sizes, then trained and evaluated a model using the selected parameters.

## Results

In the saved notebook run, **both the initial and tuned models achieved 99.81% test accuracy** on the held-out test split. The tuned model also achieved a **weighted F1 score of 99.81%**.

These are results from this dataset and split, not a claim about performance on new recordings. The unusually high accuracy prompted me to consider possible redundancy in the features and the need for more independent validation.

## Run the notebook

1. Open [`MLP_model.ipynb`](MLP_model.ipynb) in Google Colab.
2. Run the cells from top to bottom. The notebook installs Optuna and downloads the dataset from its GitHub repository.
3. See [`requirements.txt`](requirements.txt) for the main Python packages used.

The notebook loads a Python pickle file from the linked data repository, so run it only if you trust that source.

## Technologies

Python · TensorFlow/Keras · scikit-learn · NumPy · Optuna · Matplotlib · Seaborn
