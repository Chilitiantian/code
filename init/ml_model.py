from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

class Model(object):
    def __init__(self, way):
        if way == "svm":
            # Standardization = StandardScaler()
            svr = SVR(
                        kernel = "rbf", degree = 3, gamma = 0.01, 
                        coef0 = 0.0, tol = 0.001, C = 10, 
                        epsilon = 0.1, shrinking = True, cache_size = 200, 
                        verbose = False, max_iter = 1000
                    )
            # svm_estimators = [("scale",Standardization), ('svr',svr)]
            # self.way = Pipeline(steps = svm_estimators)
            self.way = svr

        if way == "rf":
            # MinM = MinMaxScaler(feature_range = (0, 1), copy = True)
            rf = RandomForestRegressor(
                            n_estimators = 10, criterion = "mae", max_depth = None,
                            min_samples_split = 2, min_samples_leaf = 1, 
                            min_weight_fraction_leaf = 0.0, max_features="auto", 
                            max_leaf_nodes = None, bootstrap = True, 
                            oob_score = False, n_jobs = 1,random_state = None,
                            verbose = 0, warm_start = False
                        )
            # rf_estimators = [("normolize",MinM), ("rf",rf)]
            # self.way = Pipeline(steps = rf_estimators)
            self.way = rf
        if way == "gbdt":
            # MinM = MinMaxScaler(feature_range = (0, 1), copy = True)
            gbdt = GradientBoostingRegressor(
                                loss = "ls", 
                                n_estimators = 100, 
                                learning_rate = 0.06, 
                                subsample = 0.4, 
                                max_depth = 20,
                            )
            # gbdt_estimators = [("normolize",MinM), ("gbdt",gbdt)]
            # self.way = Pipeline(steps = gbdt_estimators)
            self.way = gbdt