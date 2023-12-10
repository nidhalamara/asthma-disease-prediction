
# welcome in asthma-disease-prediction



please install these requirements
```bash
pip3 install Flask
pip3 install tesorflow
pip3 install numpy 
pip3 install keras
```
please check the environment first by using the API: /test_prepare_env
if everything is ready, then the next step is to train a model at least with the API:/train_model

note that you can train as many as you want,

last step is to predict using your model that you've successfully made using the API: /predict

note that the result that you've made will be stored in the dataset as a new record ,

thus when you train a new model, it will have some better results

the test body should match the same fields as in the file called schema.json


this project was made for educational purposes and was made with love.

the dataset that we've used is from https://www.kaggle.com/code/mohansss/asthma-disease-prediction/input
