from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense  # sparse or dense

class trainer:
    def __init__(self, args, data):
        self.data = data
        self.args = args

    def train(self):
        ret = {}
        self.data=self.data.dropna()
        target = 'Difficulty-in-Breathing'
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y.values.reshape(-1, 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.args.test_size,
                                                            random_state=self.args.random_state)
        y_mean = y_train.mean()
        y_pred_baseline = [y_mean] * len(y_train)
        model = Sequential()
        model.add(Dense(10, activation='relu', input_dim=18))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        model.fit(X_train, y_train, epochs=self.args.epochs, validation_split=self.args.validation_split)
        y_pred = model.predict(X_test)
        test_acc = r2_score(y_test, y_pred)
        ret['r2-score'] = test_acc
        ret['mean absolute error'] = mean_absolute_error(y_train, y_pred_baseline)
        classifier = Sequential()
        classifier.add(Dense(units=4, kernel_initializer='he_uniform', activation='relu', input_dim=18))
        classifier.add(Dense(units=4, kernel_initializer='he_uniform', activation='relu'))
        classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation="relu"))
        classifier.compile(optimizer='Adamax', loss="binary_crossentropy", metrics=["mae"])
        return classifier, ret
