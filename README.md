# Cyber Threat Pattern Classification Using Machine Learning

This project focuses on the classification and prediction of cyber threat patterns using machine learning techniques. The primary goal is to leverage the capabilities of multiple machine learning models to analyze network traffic data and identify malicious activities. The models used in this project include Random Forest, LightGBM, and Voting Classifier, with performance evaluated using metrics like accuracy, precision, recall, and cross-validation scores.

## Table of Contents

- [Project Overview](#project-overview)
- [Tools and Technologies](#tools-and-technologies)
- [Dataset](#dataset)
- [Model Selection](#model-selection)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Project Overview

The goal of this project is to classify network traffic data into benign or malicious categories to detect cyber threats in real time. The dataset used includes network traffic features, and machine learning models are trained to identify patterns that indicate attacks such as DDoS, malware, and other cyber threats.

Multiple models were implemented, including Random Forest, LightGBM, and Voting Classifier, to ensure robustness and compare the performance of various techniques in predicting cyber threats.

## Tools and Technologies

- **Programming Language**: Python
- **Libraries**: 
  - `scikit-learn`: For machine learning algorithms
  - `LightGBM`: For gradient boosting machine models
  - `pandas`: For data manipulation
  - `matplotlib` and `seaborn`: For visualization
  - `numpy`: For numerical operations
- **Environment**: Jupyter Notebook or Python IDE

## Dataset

The dataset used in this project is a network traffic dataset consisting of features such as packet lengths, flow durations, and traffic volume. The dataset contains both benign and malicious traffic, with labels indicating the type of activity. It is preprocessed to handle missing values, categorical encoding, and feature scaling, followed by dimensionality reduction using PCA.
Link to the dataset: [https://www.kaggle.com/datasets/dhoogla/cicidscollection?resource=downlo](https://www.kaggle.com/datasets/dhoogla/cicidscollection?resource=download)

## Model Selection

The project uses three primary machine learning models for classification:
1. **Random Forest**: An ensemble learning method using multiple decision trees to make predictions.
2. **LightGBM**: A gradient boosting framework that improves the speed and accuracy of models, particularly for large datasets.
3. **Voting Classifier**: A model that combines the predictions of multiple base models (Random Forest, LightGBM) by taking the majority vote.

These models were selected for their effectiveness in handling large datasets and their ability to capture non-linear relationships in the data, making them suitable for the complex nature of cyber threat detection.

## Methodology

### Data Preprocessing
The dataset is first preprocessed by removing irrelevant features, handling missing values, and performing feature scaling. A feature selection method like Variance Threshold or Recursive Feature Elimination (RFE) is applied to retain the most relevant features. The data is then split into training and testing sets, with cross-validation used to evaluate the performance of models.

### Model Training and Tuning
Each model is trained on the preprocessed data using a training set, with hyperparameters fine-tuned using techniques like GridSearchCV or RandomizedSearchCV. The models are evaluated based on their performance on the testing set.

### Handling Imbalanced Data
Since the dataset may have an imbalanced distribution of classes (malicious vs. benign), techniques such as class weighting, SMOTE (Synthetic Minority Over-sampling Technique), or undersampling can be applied to mitigate bias.

## Results

After evaluating the models, the following results were observed:

- **Random Forest**: Performance improves with more estimators and deeper trees, reaching a maximum accuracy of 78.67%.
- **LightGBM**: The accuracy stabilized around 60%, but the model showed good performance in certain scenarios where fine-tuned.
- **Voting Classifier**: Combined the strengths of Random Forest and LightGBM, achieving an accuracy of 78.67%.

Cross-validation showed that the models performed consistently, though Random Forest and Voting Classifier were the most accurate.

## Installation

To run this project locally, ensure you have Python 3.6+ installed along with the required libraries. You can install the dependencies via `pip`:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/cyber-threat-pattern-classification.git

2. Navigate to the project directory:   
   ```bash
   git clone https://github.com/your-username/cyber-threat-pattern-classification.git

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook

4. Open the notebook and execute the cells to see the models' implementation and results.

## Conclusion

This project demonstrates the use of machine learning models, including Random Forest, LightGBM, and Voting Classifier, for classifying cyber threat patterns. The models were evaluated based on accuracy, precision, recall, and cross-validation scores. The Voting Classifier performed best, providing a robust solution for threat detection. Future work could involve using deeper models such as neural networks, exploring more feature engineering techniques, or applying the models to real-time data for cybersecurity applications.
