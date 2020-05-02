import numpy as np
import pandas as pd

from IPython.display import Image 
from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 7, 5
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_style('white')


from sklearn.tree import DecisionTreeRegressor as DTR, DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.datasets import make_regression, make_classification, make_circles
from sklearn.metrics import mean_squared_error as MSE, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split, KFold

import xgboost as xgb
