from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, kernels, GaussianProcessClassifier

REGRESSION_MODELS = [
    # Support Vector Regression - RBF Kernel
    {
        'ModelClass': SVR,
        'model_params': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.1, 0.2, 0.5],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['rbf'],

        }
    },

    # Support Vector Regression - Linear Kernel
    {
        'ModelClass': SVR,
        'model_params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'kernel': ['linear'],
        }
    },


    # Support Vector Regression - Polynomial Kernel
    {
        'ModelClass': SVR,
        'model_params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
            'degree': [2, 3, 4],
            'kernel': ['poly'],
        }
    },

    # Support Vector Regression - Sigmoid Kernel
    {
        'ModelClass': SVR,
        'model_params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
            'coef0': [0, 0.1, 1],
            'kernel': ['sigmoid'],
        }
    },

    # Ridge Regression
    {
        'ModelClass': Ridge,
        'model_params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
            'fit_intercept': [True, False],
        }
    },

    # Lasso Regression
    {
        'ModelClass': Lasso,
        'model_params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False],
            'max_iter': [1000, 2000, 5000],
            'selection': ['cyclic', 'random'],
        }
    },

    # Elastic Net
    {
        'ModelClass': ElasticNet,
        'model_params': {
            'alpha': [0.001, 0.01, 0.1, 1, 10],
            'l1_ratio': [0.1, 0.2, 0.4, 0.5, 0.7, 0.9],
            'fit_intercept': [True, False],
            'selection': ['cyclic', 'random'],
        }
    },

    # Random Forest Regressor
    {
        'ModelClass': RandomForestRegressor,
        'model_params': {
            'n_estimators': [10, 20, 50, 100, 200, 300, 500, 1000],
            'max_depth': [None, 1, 2, 4, 5, 10],
            'bootstrap': [True, False],
        }
    },

    # Gradient Boosting Regressor
    {
        'ModelClass': GradientBoostingRegressor,

        'model_params': {
            'n_estimators': [10, 50, 100, 200, 300, 500],
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0],
        }
    },

    # K-Nearest Neighbors
    {
        'ModelClass': KNeighborsRegressor,

        'model_params': {
            'n_neighbors': [1, 2, 3, 5],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],  # 1: Manhattan distance, 2: Euclidean distance
        }
    },

    # Decision Tree Regressor
    {
        'ModelClass': DecisionTreeRegressor,
        'model_params': {
            'max_depth': [None, 5, 10, 20, 30, 50, 100],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
        }
    },

    # Multi-layer Perceptron (Neural Network)
    {
        'ModelClass': MLPRegressor,

        'model_params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['logistic', 'relu'],
            'solver': ['lbfgs', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1],
        }
    },

    # Gaussian Process Regressor
    {
        'ModelClass': GaussianProcessRegressor,
        'model_params': {
            'kernel': [kernels.RBF(), kernels.Matern(), kernels.RationalQuadratic(), kernels.DotProduct(), kernels.ConstantKernel()],
            'alpha': [1e-10, 1e-5, 1e-2, 1e0],
        }
    },
]


CLASSIFICATION_MODELS = [
    {
        'ModelClass': SVC,
        'model_params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'],
        }
    },
    {
        'ModelClass': LogisticRegression,
        'model_params': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2', 'l1'],
            'solver': ['liblinear'],
        }
    },
    {
        'ModelClass': RidgeClassifier,
        'model_params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True],
        }
    },
    {
        'ModelClass': RandomForestClassifier,
        'model_params': {
            'n_estimators': [10, 20, 50, 100, 200, 300, 500, 1000],
            'max_depth': [None, 1, 2, 4, 5, 10],
            'bootstrap': [True, False],
        }
    },
    {
        'ModelClass': GradientBoostingClassifier,
        'model_params': {
            'n_estimators': [10, 50, 100, 200, 300, 500],
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.8, 0.9, 1.0],
        }
    },
    {
        'ModelClass': KNeighborsClassifier,
        'model_params': {
            'n_neighbors': [1, 2, 3, 5],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],  # 1: Manhattan distance, 2: Euclidean distance
        }
    },
    {
        'ModelClass': DecisionTreeClassifier,
        'model_params': {
            'max_depth': [None, 5, 10, 20, 30, 50, 100],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
        }
    },
    {
        'ModelClass': MLPClassifier,
        'model_params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
            'activation': ['logistic', 'relu'],
            'solver': ['lbfgs', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1],
        }
    },
    {
        'ModelClass': GaussianProcessClassifier,
        'model_params': {
            'kernel': [kernels.RBF(), kernels.Matern(), kernels.RationalQuadratic(), kernels.DotProduct(), kernels.ConstantKernel()],
        }
    },
]


def is_regressor(ModelClass):
    if ModelClass in [model['ModelClass'] for model in REGRESSION_MODELS]:
        return True
    if ModelClass in [model['ModelClass'].__name__ for model in REGRESSION_MODELS]:
        return True
    if not isinstance(ModelClass, str):
        if ModelClass.__class__ in [model['ModelClass'] for model in REGRESSION_MODELS]:
            return True
    if ModelClass in [model['ModelClass'] for model in CLASSIFICATION_MODELS]:
        return False
    if ModelClass in [model['ModelClass'].__name__ for model in CLASSIFICATION_MODELS]:
        return False
    if not isinstance(ModelClass, str):
        if ModelClass.__class__ in [model['ModelClass'] for model in CLASSIFICATION_MODELS]:
            return False
    raise ValueError(f"ModelClass {ModelClass} is not recognized as a regressor or classifier.")


MODEL_CLASSES = {
    model['ModelClass'].__name__: model['ModelClass']
    for model in REGRESSION_MODELS + CLASSIFICATION_MODELS
}
