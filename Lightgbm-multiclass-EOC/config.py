class Config(object):

    def __init__(self):
        self.test_size = 0.3
        self.random_state = 123
        self.model_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_class': 4,
            'num_leaves': 12,
            'learning_rate': 0.035,
            'bagging_seed': 42,
            'feature_fraction_seed': 42,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'force_col_wise': False,
            'min_data_in_leaf': 28,
            'verbose': -1,
            'max_depth': 6,
        }


    def get_model_params(self):
        return self.model_params
