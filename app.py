from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
filename = 'wind.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)

@app.route('/')


def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():
    RAIN = request.form['RAIN']
    T_MIN = request.form['T_MIN']
    T_MAX = request.form['T_MAX']
    T_MIN_G = request.form['T_MIN_G']
    IND = request.form['IND']
    IND_1 = request.form['IND_1']
    IND_2 = request.form['IND_2']

    pred = model.predict(np.array([[RAIN,T_MIN,T_MAX,T_MIN_G,IND,IND_1,IND_2]], dtype=float))

    return render_template('index.html', predict=str(pred))

# if __name__ == '__main__':
#     app.run

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
