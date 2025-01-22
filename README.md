# WATER QUALITY ML
This project is a Water Quality Detection Algorithm built using Python and machine learning. The algorithm predicts water quality as Good, Average, or Poor based on various water parameters. It uses a dataset of water quality measurements, processes it, and trains a Random Forest Classifier to classify water quality. The project also supports predicting water quality for new datasets and saving the results.

# Features
Encodes categorical features like State Name and Type of Water Body.
Classifies water quality into three categories: Good, Average, or Poor.
Employs Random Forest Classifier for high-accuracy predictions.
Supports applying the trained model to new datasets and saving predictions to CSV files.
Saves the trained model and encoders for reuse.

# Project Workflow
1. Data Preparation
The dataset includes water quality parameters such as:
pH
Temperature
Dissolved Oxygen
Conductivity
BOD (Biochemical Oxygen Demand)
Nitrate/Nitrite Levels
Fecal Coliform Levels
Categorical features (State Name and Type of Water Body) are label-encoded using sklearn.preprocessing.LabelEncoder.

2. Water Quality Categorization
The water quality is categorized using custom rules:
Good: Strict thresholds for all water quality parameters.
Average: Slight deviations from ideal thresholds.
Poor: Significant deviations.

3. Model Training
The processed data is split into training and testing sets.
A Random Forest Classifier is trained on the dataset.

4. Evaluation
The trained model is evaluated using:
Accuracy Score
Classification Report
Confusion Matrix

5. Predicting New Data
The model predicts water quality for new datasets.
Results are saved as a CSV file for further analysis.

# How to use

1. Install required python libraries
2. Make sure the columns of the dataset that you want to use matches the columns of the dataset given in the project.
3. Open terminal and use command - python quality-of-water-bodies-eda.ipynb
4. The model should now be saved in the folder which will be used in the application.
5. Use command - python app.py
6. The application will automatically start running on a local host server.

# Technologies Used
Python
Pandas for data manipulation
NumPy for numerical computations
scikit-learn for machine learning and preprocessing
joblib for saving and loading models
CSV for handling datasets
