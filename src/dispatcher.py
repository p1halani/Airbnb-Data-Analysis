from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from skopt.space import Real
from sklearn.tree import DecisionTreeRegressor

0.75091
regr = GradientBoostingRegressor(criterion = 'friedman_mse', random_state=42)
param_dist = {"max_depth": stats.randint(2, 6),
              "n_estimators": stats.randint(50, 200)}
#               "learning_rate": Real(10**-1, 10**0, "log-uniform", name='learning_rate'),
#              "subsample": stats.uniform(0.5, 0.5)}
n_iter_search = 50
random_search = RandomizedSearchCV(regr, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=3, verbose=2)



MODELS = {
    "gradientboost": random_search,
    "decisiontree": DecisionTreeRegressor(random_state=42,
                                            criterion='friedman_mse'),
    "randomforest": ensemble.RandomForestRegressor(n_estimators=200,
                                                    n_jobs=4,
                                                    random_state=42,
                                                    verbose=2,
                                                    criterion='friedman_mse'),
    "extratrees": ensemble.ExtraTreesRegressor(n_estimators=200,
                                                n_jobs=4,
                                                random_state=42,
                                                criterion='friedman_mse',
                                                verbose=2),
}