# Skin Cancer Detection
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Table of Contents
* [Dataset](#dataset)
* [CNN Design](#cnn-design)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
# Dataset
The dataset has been uploaded to Kagglge - https://www.kaggle.com/datasets/kk20krishna/skin-cancer-data-isic

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

## CNN Design
**Cancer Classification Model Design**

**Model Architecture**

This Convolutional Neural Network (CNN) is designed to classify images into 9 distinct cancer classes. The input images have a resolution of 180x180 pixels with 3 color channels (RGB).

Below, the design choices made for this model are explained:

1. **Input Preprocessing**
  - **Rescaling:** The input pixel values are normalized to the range [0,1] using `Rescaling(scale=1./255)`. This helps in stabilizing the learning process by ensuring smaller gradients during training.

2. **Feature Extraction using Convolutional Blocks**
The model consists of four convolutional blocks, each having:
      1. **Conv2D Layer**: Extracts spatial features using 3x3 filters with 'same' padding to maintain spatial dimensions.
      2. **ReLU Activation**: Introduces non-linearity, allowing the model to learn complex patterns.
      3. **MaxPooling Layer**: Reduces spatial dimensions and computational cost while preserving important features.

  - **First Convolutional Block:**
    - 32 filters, 3x3 kernel, ReLU activation, 'same' padding.
    - Followed by MaxPooling to reduce spatial size.

  - **Second Convolutional Block:**
    - 64 filters, 3x3 kernel, ReLU activation, 'same' padding.
    - Followed by MaxPooling.

  - **Third Convolutional Block:**
    - 128 filters, 3x3 kernel, ReLU activation, 'same' padding.
    - Followed by MaxPooling.

  - **Fourth Convolutional Block:**
    - 256 filters, 3x3 kernel, ReLU activation, 'same' padding.
    - Followed by MaxPooling.

3. **Regularization using Dropout**
  - A **Dropout layer (50%)** is introduced after the fourth convolutional block to reduce overfitting by randomly setting neuron outputs to zero during training.

4. **Fully Connected Layers**
  - **Flatten Layer**: Converts the feature maps into a 1D vector.
  - **Dense Layer with 128 neurons (ReLU activation)**: Helps in learning complex feature representations.
  - **Dropout Layer (50%)**: Provides additional regularization.

5. **Output Layer**
  - **Dense Layer with 9 neurons**: Uses Softmax activation to output probabilities for the 9 cancer classes.

![image](https://github.com/user-attachments/assets/589c00da-e0fe-4c07-bf48-cb7a5abbbba3)


## Conclusions
- Conclusion 1 from the analysis
- Conclusion 2 from the analysis
- Conclusion 3 from the analysis
- Conclusion 4 from the analysis

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- library - version 1.0
- library - version 2.0
- library - version 3.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.
- This project was inspired by...
- References if any...
- This project was based on [this tutorial](https://www.example.com).


## Contact
Created by [@githubusername] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
