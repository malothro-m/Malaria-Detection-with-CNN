# Malaria Detection using Convolutional Neural Networks (CNN)

This project is focused on developing a machine learning model for detecting malaria from blood smear images using Convolutional Neural Networks (CNN). The model is built using TensorFlow and is trained on a dataset containing parasitized and uninfected cell images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Project Overview
Malaria is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. This project aims to automate the detection of malaria using CNNs to analyze cell images, thereby assisting in rapid and accurate diagnosis.

## Dataset
The dataset used in this project consists of images of blood smears that are categorized into two classes:
1. **Parasitized** - Images showing the presence of malaria parasites.
2. **Uninfected** - Images that are free from malaria parasites.

### Data Source
- The dataset is available on Kaggle: [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria).

## Model Architecture
The CNN model consists of several layers:
- **Conv2D Layers**: Three convolutional layers with 16, 32, and 64 filters respectively, each followed by a ReLU activation function.
- **MaxPooling2D Layers**: Three max pooling layers to downsample the feature maps.
- **Flatten Layer**: Converts the 2D feature maps into a 1D feature vector.
- **Dense Layers**: Two dense layers, with the first one having 64 units and ReLU activation, and the second one with a single unit and sigmoid activation for binary classification.
- **Dropout Layer**: A dropout layer with a rate of 0.35 to prevent overfitting.

## Data Preprocessing
- **Loading Data**: The images are loaded from the local directory and preprocessed using the `load_data` function.
- **Image Resizing**: Images are resized to 64x64 pixels.
- **Normalization**: The pixel values are normalized to the range [0, 1].
- **Augmentation**: Training data is augmented with transformations like rotation, shifting, shearing, zooming, and horizontal flipping.

## Training and Evaluation
- The dataset is split into training (75%) and testing (25%) sets.
- The model is trained for 20 epochs with `binary_crossentropy` as the loss function and `adam` as the optimizer.
- Evaluation is performed on the testing set to determine the accuracy of the model.

## Results
The model achieved a test accuracy of **X.XX**% after training. The training and validation accuracy are plotted for each epoch to visualize the model's performance over time.

## Dependencies
- TensorFlow
- TensorFlow Datasets
- Matplotlib

You can install the required packages using the following command:
```bash
pip install tensorflow tensorflow-datasets matplotlib

