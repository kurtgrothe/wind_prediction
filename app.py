from flask import Flask, render_template, request
import numpy as np
import torch
# import pickle
# import joblib
import os
import torch
import torch.nn as nn

# ANN Model required for PyTorch prediction
class ANN(nn.Module):
    def __init__(self, input_shape):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 512 * input_shape)
        self.fc2 = nn.Linear(512 * input_shape, 256 * input_shape)
        self.fc3 = nn.Linear(256 * input_shape, 128 * input_shape)
        self.fc4 = nn.Linear(128 * input_shape, 64 * input_shape)
        self.fc5 = nn.Linear(64 * input_shape, 1)
        self.elu = nn.ELU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.elu(self.fc5(x))
        return x

app = Flask(__name__, template_folder='templates', static_folder='static')

filename = 'data\wind.pt'
input_shape = 7
model = torch.load(open(filename, 'rb'))
model_state_dict = torch.load(open(filename, 'rb'))
model = ANN(input_shape)
model.load_state_dict(model_state_dict)

# filename = 'data\wind.pkl'
# model = joblib.load(filename)

@app.route('/')


def index():
    return render_template('index.html')

# PyTorch prediction function
@app.route('/predict', methods=['POST'])
def predict():
    RAIN = float(request.form['RAIN'])
    T_MIN = float(request.form['T_MIN'])
    T_MAX = float(request.form['T_MAX'])
    T_MIN_G = float(request.form['T_MIN_G'])
    IND = float(request.form['IND'])
    IND_1 = float(request.form['IND_1'])
    IND_2 = float(request.form['IND_2'])

    input_data = torch.tensor([[RAIN, T_MIN, T_MAX, T_MIN_G, IND, IND_1, IND_2]], dtype=torch.float32)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output = model(input_data)  # Run the input data through the model

    pred = output.item()  # Convert the output tensor to a single scalar value

    return render_template('index.html', predict=str(pred))

# # Scikit-learn prediction function
# @app.route('/predict', methods=['POST'])

# def predict():
#     RAIN = request.form['RAIN']
#     T_MIN = request.form['T_MIN']
#     T_MAX = request.form['T_MAX']
#     T_MIN_G = request.form['T_MIN_G']
#     IND = request.form['IND']
#     IND_1 = request.form['IND_1']
#     IND_2 = request.form['IND_2']

#     pred = model.predict(np.array([[RAIN,T_MIN,T_MAX,T_MIN_G,IND,IND_1,IND_2]], dtype=float))

#     return render_template('index.html', predict=str(pred))

# if __name__ == '__main__':
#     app.run

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
