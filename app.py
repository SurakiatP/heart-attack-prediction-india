from flask import Flask, request, jsonify
import pickle
import os

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route("/", methods=["GET"])
def home():
    return "Hello! Your Flask server is running."

# Define the path to the model file (the model should be saved during training)
model_path = os.path.join("models", "rf_model.pkl")
# Check if the model file exists; if not, raise an error
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

# Load the trained model from the pickle file
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expect a JSON payload from the client in the following format:
    {
        "features": [[feature1, feature2, ...], [feature1, feature2, ...], ...]
    }
    Alternatively, it could be a list of dictionaries if you adjust the code accordingly.
    """
    # Get JSON data from the POST request
    data = request.get_json()
    # Use the loaded model to make predictions on the provided features
    prediction = model.predict(data["features"])
    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction.tolist()})

# Run the Flask app on host 0.0.0.0 at port 8000 when executed directly
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
