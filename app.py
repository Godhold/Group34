from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Load the pre-trained model and scaler
model = joblib.load('sports_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# ... (previous code)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input data from the web form
        input_data = request.form.to_dict()

        # Convert input data to a DataFrame
        input_data = pd.DataFrame(input_data, index=[0])

        # Preprocess and scale the input data using the scaler
        scaled_input = scaler.transform(input_data)

        # Use the model to make predictions
        prediction = model.predict(scaled_input)[0]

        # Return the prediction result to the web page
        return render_template('index.html', prediction=f'Predicted Overall Rating: {prediction:.2f}')
    except Exception as e:
        return str(e)

# ... (remaining code)

if __name__ == '__main__':
    app.run(debug=True)