from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Load the trained model and feature columns
loaded_model = joblib.load('gradient_boosting_f1_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form submission
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])

        # Convert numeric fields
        for col in ['Points', 'Laps', 'Season', 'Points_Laps', 'Points_Season', 'Best_Position_Points']:
            input_df[col] = pd.to_numeric(input_df[col])

        # Preprocess the input
        input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=feature_columns, fill_value=0)

        # Make predictions
        prediction = loaded_model.predict(input_encoded.to_numpy())[0]

        # Return the result
        return render_template('index.html', prediction=round(prediction, 2))
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)