import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LightGBMDataLoader:
    def __init__(self, data_path, target_column, config):
        self.data_path = data_path
        self.target_column = target_column
        self.config = config

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def load_data(self):
        data = pd.read_csv(self.data_path)

        data = data.drop(data.columns[0], axis=1)

        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]

        # Replaces special characters with underscores
        X.columns = [feature_name.replace('%', '_').replace('(', '_').replace(')', '_').
                     replace('.', '_').replace(',', '_').lstrip('_') for feature_name in X.columns]

        # Train_Test_Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.config.test_size,
                                                                                random_state=self.config.random_state)

    def preprocess_data(self):
        scaler = StandardScaler()

        self.X_train_scaled = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)

        self.X_test_scaled = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)

        return self.X_train_scaled, self.X_test_scaled

    def load_and_preprocess_data(self):
        self.load_data()
        self.preprocess_data()