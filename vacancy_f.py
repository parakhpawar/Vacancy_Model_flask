import pickle
from flask import Flask, request

# create a Flask app
app = Flask(__name__)

# load the pickled model
with open('vacancy_flask.pkl', 'rb') as f:
    model = pickle.load(f)

# define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    # get the request data
    data = request.get_json()

    # make a prediction using the loaded model
    prediction = model.predict(data)

    # return the prediction as a JSON response
    return {'prediction': prediction.tolist()}

# start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
