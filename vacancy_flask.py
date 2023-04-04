from flask import Flask , request, jsonify
import numpy as np
import pickle

model = pickle.load(open('vacancy_flask.pkl','rb'))
print("model is loaded")


app= Flask(__name__)

@app.route('/')
def index():
   num_rooms_available =  int(request.args['num_rooms_available'])
   num_rooms_booked = int(request.args['num_rooms_booked'])
   num_cancellations = int(request.args['num_cancellations'])
   avg_length_of_stay = int(request.args['avg_length_of_stay'])
   location_City =int(request.args['location_City Hotel'])
   location_Resort = int(request.args['location_Resort Hotel'])

   pred = model.predict(np.array(['num_rooms_available','num_rooms_booked','num_cancellations','avg_length_of_stay','location_City Hotel','location_Resort Hotel']).reshape(1,-1))


   return jsonify(prediction = str(pred))

if __name__  ==  "__main__":
    app.run(debug=True)
