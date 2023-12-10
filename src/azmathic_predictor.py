from keras.models import load_model
import pandas as pd
import json


class Azmathic_Predictor:
    def __init__(self, args):
        self.args = args
        self.model = None

    def check_capability(self, test_input):
        ret = {}
        try:
            with open("src/ressources/schema.json", 'r') as file:
                schema = json.load(file)
                user_schema = test_input.keys()
                for i in user_schema:
                    if i not in schema:
                        raise AssertionError
        except:
            ret['ERROR'] = "WHEATHER SCHEMA NOT FOUND OR SOME OF YOUR VALS ARE MISSING"
            return ret

    def normalize_data(self, test_input):
        normalized_data = {'Tiredness': 1 if test_input['Tiredness'] == 'true' else 0,
                           'Dry-Cough': 1 if test_input['Dry-Cough'] == 'true' else 0,
                           # 'Difficulty-in-Breathing': 1 if test_input['Difficulty-in-Breathing'] == "true" else 0,
                           'Sore-Throat': 1 if test_input['Sore-Throat'] == 'true' else 0,
                           'None_Sympton': 1 if test_input['None_Sympton'] == 'true' else 0,
                           'Pains': 1 if test_input['Pains'] == 'true' else 0,
                           'Nasal-Congestion': 1 if test_input['Nasal-Congestion'] == 'true' else 0,
                           'Runny-Nose':1 if test_input['Runny-Nose']=='true'else 0,
                           'None_Experiencing':1 if test_input['None_Experiencing']=='true'else 0,
                           'Age_0-9': 1 if test_input['Age'] < 10 else 0,
                           'Age_10-19': 1 if 10 <= test_input['Age'] < 20 else 0,
                           'Age_20-24': 1 if 20 <= test_input["Age"] < 25 else 0,
                           'Age_25-59': 1 if 25 <= test_input["Age"] < 60 else 0,
                           'Age_60+': 1 if test_input['Age'] >= 60 else 0,
                           'Gender_Male': 1 if test_input['Gender'] == "Male" else 0,
                           'Gender_Female': 1 if test_input['Gender'] == 'Female' else 0,
                           'Severity_Mild': 1 if test_input['Severity'] == 'Mild' else 0,
                           'Severity_Moderate': 1 if test_input['Severity'] == 'Moderate' else 0,
                           'Severity_None': 1 if test_input['Severity'] == 'None' else 0}
        return normalized_data

    def predict(self, test_input):
        ret = {}
        if self.args.model_path == "" or None:
            ret['ERROR'] = "NO WORKING PATH WAS PROVIDED \n, WHEATHER YOUR MODEL IS TRAINING \nOR\n YOU MAY SHOULD " \
                           "RUN THE TRAIN API TO TRAIN A MODEL FIRST "
            return ret
        else:
            try:
                self.check_capability(test_input)
                self.model = load_model(self.args.model_path)
                try:
                    test_data = self.normalize_data(test_input)
                except Exception as e:
                    ret['ERROR'] = f'ERROR IN NORMALIZING YOUR INPUT ,,LOG: {e}'
                    return ret
                result=self.model.predict(pd.DataFrame([test_data]))[0].tolist()[0]
                ret['SUCC'] = 1 if result>0.4 else 0
                return ret
            except Exception as e:
                ret['ERROR'] = f'WEATHER YOUR MODEL WAS NOT LOADED CORRECTLY OR : LOG: {e}'
                return ret
