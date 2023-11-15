import lightgbm as lgb


def train_lightgbm_model(X_train_scaled, y_train, model_params):
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    model = lgb.train(model_params, train_data, num_boost_round=100)
    return model
