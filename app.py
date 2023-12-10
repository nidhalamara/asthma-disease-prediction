import datetime

from flask import Flask, request
from src.args import args as ar
from src.trainer import trainer as tr
from src.azmathic_predictor import Azmathic_Predictor as pr
import os
import json

import pandas as pd

app = Flask(__name__)


@app.route('/test_prepare_env', methods=['GET'])
def test_prepare_env():
    try:
        open("src/ressources/processed-data.csv")
    except:
        return {"ERROR": "NO DATA FILE TO TRAIN MODEL, PLEASE CONTACT THE ADMINISTRATOR"}
    return {"SUCC": "EVERYTHING IS FINE , TRAIN YOUR MODEL"}


@app.route('/train_model', methods=['GET'])
def prepare_model():
    args = ar(1, 0.2, 20, 0.2, "src/ressources/processed-data.csv")
    d = pd.read_csv("src/ressources/processed-data.csv")
    t = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    args.model_path = "src/models/model-" + str(t) + ".h5"
    trainer = tr(args, d)
    try:
        model, r = trainer.train()
        model.save(args.model_path)
        with open("db/models.json", 'r') as f:
            data = json.load(f)
            data[str(t)] = args.model_path
        with open("db/models.json", 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        return {"ERROR": f"PLEASE CONTACT THE ADMIN , COMPLETE LOG: {e}"}
    return {"SUCC": f"YOUR MODEL IS SAVED WITH THE Name:{args.model_path}", 'time-of-training': str(t)}


@app.route('/predict', methods=['POST'])
def predict_vals():
    try:
        data = request.get_json(force=True)
        args = ar(0, 0, 0, 0, '')
        try:
            args.model_path = resolve_model_path()
        except BaseException as e:
            return {"ERROR": f"COMPLETED LOG:{e} "}
        predictor = pr(args)
    except BaseException as e:
        return {"ERROR": f"COMPLETE LOG:{e} "}
    return predictor.predict(data)


def resolve_model_path():
    json_file_path = os.path.join('db', 'models.json')
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            dates = [i for i in data.keys()]
            try:
                path = sorted(dates)[0]
                return data[path]
            except BaseException("NO MODELS EXIST, PLEASE TRAIN ONE MODEL AT LEAST") as e:
                return e
    else:
        raise BaseException("NO PATH PROVIDED IS WORKING , PLEASE CONTACT THE ADMIN")
    return None


@app.route('/')
def hello_world():  # put application's code here
    return '''
    welcome in asthma-disease-prediction \n
    please check the environment first by using the API: /test_prepare_env\n
    if everything is ready, then the next step is to train a model at least with the API:/train_model \n
    note that you can train as many as you want,\n
    last step is to predict using your model that you've successfully made using the API: /predict\n
    note that the result that you've made will be stored in the dataset as a new record ,\n
    thus when you train a new model, it will have some better results\n
    \n
    \n
    this project was made for educational purposes : \n
    the dataset that we've used is from https://www.kaggle.com/code/mohansss/asthma-disease-prediction/input\n
    this project is made with love by :\n
      \t\t      -Nidhal Amara\n
       \t\t     -Oumayma samoudi\n
      \t\t      -Ghaith mefteh\n
    '''


if __name__ == '__main__':
    app.run(debug=True)
