# Skin Cancer Detection
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


## Table of Contents
* [Dataset](#dataset)
* [CNN Design](#cnn-design)
* [Conclusions](#conclusions)
* [Technologies Used](#technologies-used)
* [Acknowledgements](#acknowledgements)

## General Information
# Dataset
The dataset has been uploaded to Kagglge - https://www.kaggle.com/datasets/kk20krishna/skin-cancer-data-isic

The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following 9 diseases:

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

Convolutional Neural Network (CNN) designed to classify images into 9 distinct cancer classes. The input images have a resolution of 180x180 pixels with 3 color channels (RGB).
The model classifies images into 9 classes.

The design choices made for this model are explained in the notebook.

![image](https://github.com/user-attachments/assets/a2293089-6337-46c5-a89c-e6d951512291)
> Image created using https://alexlenail.me/NN-SVG/AlexNet.html



![image](https://github.com/user-attachments/assets/589c00da-e0fe-4c07-bf48-cb7a5abbbba3)



## Conclusions

**1. Overview**
This report provides an evaluation of the Convolutional Neural Network (CNN) model trained for melanoma detection and cancer classification. The evaluation metrics include accuracy, precision, recall, F1-score, and loss for both training and validation datasets.

**2. Performance Metrics**

- **Training Performance**
  - Final Training Accuracy: 90.08%
  - Final Training F1-score: 89.83%
  - Final Training Precision: 91.81%
  - Final Training Recall: 88.71%
  - Final Training Loss: 0.2508

- **Validation Performance**
  - Best Validation Accuracy: 81.82%
  - Final Validation Accuracy: 81.47%
  - Final Validation F1-score: 80.88%
  - Final Validation Precision: 83.06%
  - Final Validation Recall: 80.22%
  - Final Validation Loss: 0.7534

**3. Training vs Validation Trends**

- **Accuracy**
  - Training accuracy remained consistently above 89%.
  - Validation accuracy fluctuated between 81% and 82%, peaking at 81.82%.
  - The gap between training and validation accuracy suggests minor overfitting.

- **F1-score**
  - Training F1-score improved steadily, reaching 89.83%.
  - Validation F1-score stabilized around 80.88%, indicating the model maintains good balance in classification.

- **Loss Analysis**
  - Training loss steadily decreased to **0.2508**, showing good convergence.
  - Validation loss plateaued around **0.75**, indicating room for further generalization improvements.


- **4. Learning Rate Analysis**
  - The final learning rate decay suggests the model was approaching optimal convergence.


-  **5. Confusion Matrix**

![image](https://github.com/user-attachments/assets/f00f1c19-1044-4d6c-ba19-5ff1e5252057)

  -  **Key Takeaways**
      -  'Nevus' and 'Melanoma' have the highest confusion. This makes sense since melanoma can visually resemble a nevus.
      -  'Actinic Keratosis' is often confused with 'Squamous Cell Carcinoma' and 'Nevus.'
      -  'Vascular Lesion' and 'Seborrheic Keratosis' show strong classification performance.
      -  'Melanoma' misclassification is concerning since early detection is critical for treatment.

- **6. Observations**
  - **Slight Overfitting**: The training accuracy is higher than validation accuracy, indicating overfitting. Consider applying stronger regularization techniques such as dropout, weight decay, or data augmentation.
  - **Validation Performance Plateau**: Validation accuracy remained stable around **81%**, indicating that performance improvements may require architectural modifications or additional data.
  - **Early Stopping Activation**: Training stopped at **epoch 74**, suggesting further training would not yield significant improvements.


- **7. Conclusion**
  The model achieved **90.08% training accuracy** and **81.82% best validation accuracy** for cancer detection.


## Technologies Used
- Augmentor
- shutil
- pathlib
- tensorflow
- matplotli
- numpy
- pandas
- os
- PIL
- keras
- kagglehub
- cv2 (OpenCV)
- scikit-learn
- seaborn

Additionally, various TensorFlow and Keras modules were utilized for building and training the model, including layers, models, callbacks, and regularizers.

## Acknowledgements
- Cource content in upGrad course

## Contact
Created by [@kk20krishna](https://github.com/kk20krishna)


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
