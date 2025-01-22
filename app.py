import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Use the non-GUI Agg backend for matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_file, url_for

# Load the trained models and encoders
model = joblib.load('model.pkl')
label_encoder_state = joblib.load('label_encoder_state.pkl')
label_encoder_type = joblib.load('label_encoder_type.pkl')
X_train_dtypes = joblib.load('X_train_dtypes.pkl')

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
PREDICTION_FILE = os.path.join(RESULT_FOLDER, 'predicted_water_quality.csv')

# Create upload and result folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Load and preprocess the uploaded CSV file
        new_dataset = pd.read_csv(file_path, encoding='latin1')
        new_dataset = new_dataset.replace('-', np.nan)
        new_dataset = new_dataset.replace('\n4', '', regex=True)
        new_dataset = new_dataset.replace('\n', ' ', regex=True)

        # Transform categorical columns
        new_dataset['State Name'] = label_encoder_state.transform(new_dataset['State Name'])
        new_dataset['Type Water Body'] = label_encoder_type.transform(new_dataset['Type Water Body'])

        # Ensure column order matches the trained model's input
        new_dataset = new_dataset[list(X_train_dtypes.keys())]

        # Match data types to the training data
        new_dataset = new_dataset.astype(X_train_dtypes)

        # Predict water quality
        new_dataset['Predicted Water Quality'] = model.predict(new_dataset)

        # Save the predictions to a CSV file
        new_dataset.to_csv(PREDICTION_FILE, index=False)

        return render_template('results.html', file_path=PREDICTION_FILE)

    except Exception as e:
        return f"Error processing file: {e}"
    
@app.route('/download')
def download_file():
    return send_file(PREDICTION_FILE, as_attachment=True)

@app.route('/visualize')
def visualize():
    try:
        # Load the dataset with predictions
        predicted_df = pd.read_csv(PREDICTION_FILE)

        # Reverse label encoding for 'State Name'
        predicted_df['State Name'] = label_encoder_state.inverse_transform(predicted_df['State Name'])

        # Count the number of occurrences of each water quality in each state
        state_quality_counts = predicted_df.groupby(['State Name', 'Predicted Water Quality']).size().unstack(fill_value=0)

        # Calculate the percentage of each quality class per state
        state_quality_percent = state_quality_counts.div(state_quality_counts.sum(axis=1), axis=0) * 100

        # Create the plot using a non-GUI backend
        plt.figure(figsize=(14, 8))
        state_quality_percent.plot(kind='bar', stacked=True, colormap="Set3", ax=plt.gca())

        plt.title('Water Quality Distribution by State')
        plt.ylabel('Percentage (%)')
        plt.xlabel('State')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Water Quality', loc='upper left')
        plt.tight_layout()

        # Save the plot to a file instead of displaying it
        plot_file = os.path.join(app.config['RESULT_FOLDER'], 'water_quality_distribution.png')
        plt.savefig(plot_file)
        plt.close()  # Close the plot to release resources

        # Find the state with the best water quality (highest percentage of 'Good')
        best_state = state_quality_percent['Good'].idxmax()
        best_state_percentage = state_quality_percent['Good'].max()

        # Return the visualization page with additional insights
        return render_template(
            'visualize.html',
            image_path=url_for('static', filename='results/water_quality_distribution.png'),
            best_state=best_state,
            best_state_percentage=f"{best_state_percentage:.2f}"
        )

    except Exception as e:
        return f"Error in visualization: {e}"

# Serve the 'results' folder as static files
@app.route('/results/<filename>')
def serve_result(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)
