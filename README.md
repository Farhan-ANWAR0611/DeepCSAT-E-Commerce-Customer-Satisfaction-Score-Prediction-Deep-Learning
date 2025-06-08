# DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction-Deep-Learning
# Customer Satisfaction (CSAT) Prediction Using Artificial Neural Network (ANN)

## Project Overview

This project focuses on predicting Customer Satisfaction (CSAT) scores for an e-commerce platform using a Deep Learning Artificial Neural Network (ANN) model. The CSAT score is treated as a 3-class classification problem (e.g., Low, Medium, High satisfaction).

The goal is to accurately classify customers into satisfaction categories based on their behavior and transactional data, enabling businesses to improve customer service and retention strategies.

## Dataset

- The dataset contains customer interaction and transaction features.
- The target variable is the CSAT score, categorized into three classes.
- Data preprocessing steps include handling missing values, encoding categorical variables, and feature scaling.

## Model Architecture

- A sequential ANN model with:
  - Input layer matching feature dimensions.
  - Two hidden layers (128 and 64 neurons) with ReLU activation.
  - Dropout layer to prevent overfitting.
  - Output layer with 3 neurons and softmax activation for multi-class classification.
- Compiled with Adam optimizer and categorical crossentropy loss.

## Implementation Steps

1. **Data Loading and Exploration**: Load dataset, inspect and understand features.
2. **Data Preprocessing**: Clean data, handle missing values, encode categorical variables, and normalize features.
3. **Train-Test Split**: Split the dataset into training and testing sets.
4. **Model Building**: Define and compile the ANN architecture.
5. **Model Training**: Train the model on training data with validation.
6. **Evaluation**: Evaluate model performance using accuracy, confusion matrix, and classification reports.
7. **Prediction**: Predict CSAT classes on new or test data.

## Results

- Achieved an accuracy of **XX%** on the test set.
- Confusion matrix and classification report illustrate model performance across classes.
- Model effectively predicts customer satisfaction levels to assist business decision-making.

## Tools and Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib / Seaborn for visualization

## How to Run

1. Clone this repository or download the notebook.
2. Install required libraries (listed in `requirements.txt` if provided).
3. Open the Jupyter notebook or Google Colab and run cells step-by-step.
4. Modify parameters or dataset paths as needed.

## Future Work

- Experiment with deeper or more complex neural networks.
- Use techniques like Batch Normalization or DropConnect to improve generalization.
- Deploy the model as a web app or API for real-time CSAT prediction.
- Explore feature engineering and additional data sources.


---

Feel free to reach out for any queries or collaboration!

