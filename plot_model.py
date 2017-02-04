#!/usr/bin/env python

from __future__ import print_function

import psycopg2
import getpass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import izip
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import f_regression, RFECV
from xgboost.sklearn import XGBRegressor

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

from utils import save, load, resave, ensure_dir

sns.set_style ("whitegrid", rc=dict(font_scale=1.3))

def pred_error (model, X_new, X_old, y_old):
    n = len (y_old)
    n_dof = n - X_old.shape[1] - 1

    pred_old = model.predict (X_old)

    mean_X_old = X_old.mean (axis=0)
    y_res_old = (y_old - pred_old)

    mse_y = np.sqrt ((y_res_old**2).sum () / n_dof)

    sq_err_X_old = np.sum (((X_old - mean_X_old)**2).sum (axis=1))
    sq_err_X_new = np.sum ((X_new - mean_X_old)**2)

    se_x = mse_y * np.sqrt (1 + 1. / n + sq_err_X_new / sq_err_X_old)
    return (se_x)


if __name__ == '__main__':

    plt_dir = ensure_dir ('/home/rmaunu/sandbox/insight/stroke_prev/plots')
    ml_dir = ensure_dir ('/home/rmaunu/sandbox/insight/stroke_prev/ml')

    ml_model = load ('ml/linear_lasso_model.pickle')

    X_train = load ('ml/X_train.pickle')
    X_test = load ('ml/X_test.pickle')
    y_train = load ('ml/y_train.pickle')
    y_test = load ('ml/y_test.pickle')

    for i in range (10):
        print (ml_model.predict (X_test.iloc[i, :])[0],
               pred_error (ml_model, X_test.iloc[i, :], X_train, y_train))
