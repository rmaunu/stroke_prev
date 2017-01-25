#!/usr/bin/env python

from __future__ import print_function

import psycopg2
import getpass
import numpy as np
import pandas as pd

from itertools import izip
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer

from utils import save, load, resave, ensure_dir

# For CDC data, remove (not enough data):
#     FIPS, State, County,
#     demographic specific blood pressure med. adherence,
#     hospital number, specialist number
CDC_SKIP_COLS = ['FIPS', 'County', 'State', 'htnadh_white', 'htnadh_black',
                 'htnadh_api', 'htnadh_aian', 'htnadh_hisp', 'stg_hosp',
                 'stg_int_cr', 'stg_rehab', 'stg_emerg', 'stg_neuro', 'cvd_dr',
                 'neuro_dr', 'surg_dr']


def get_db_connection (dbname, username,
                       password=None,
                       host='/var/run/postgresql'):
    """
    Get the SQL database connection.

    :type   dbname: str
    :param  dbname: Name of the database

    :type   username: str
    :param  username: username to access database

    :type   password: str
    :param  password: password to access database (default: None)

    :type   host: str
    :param  host: host of the database
        (default: '/var/run/postgresql', local PostgreSQL)

    :return: `psycopg2` connection
    """

    con = psycopg2.connect(
        database=dbname, user=username, password=password,
        host='/var/run/postgresql')
    return (con)


def get_cdc_data (connection, primary_tablename,
                  join_tablenames=[],
                  join_ons=[],
                  join_val_names=[]):
    """
    Get CDC data from the SQL database.

    :type   connection: database connection (e.g. `psycopg2`)
    :param  connection: A connection to an SQL database.

    :type   primary_tablename: str
    :param  primary_tablename: Name of the table to open.

    :type   join_tablenames: list
    :param  join_tablenames: List of tables to left-join with the primary table

    :type   join_ons: list of tuples
    :param  join_ons: list of column pairs which should be equal. First value of
        the tuple is the primary table column, second the join table value.

    :type   join_val_names: list
    :param  join_val_names: list of names to convert the 'Value' column to

    :return: `pandas.DataFrame` of the data
    """

    # build query
    sql_query = 'SELECT {0}.*'.format (primary_tablename)
    for t, name in izip (join_tablenames, join_val_names):
        sql_query += ', {0}."Value" AS {1}'.format (t, name)
    sql_query += ' FROM {0}'.format (primary_tablename)
    for t, join in izip (join_tablenames, join_ons):
        sql_query += ' LEFT JOIN {0} ON {1}.{2}={0}.{3}'.format (
            t, primary_tablename, join[0], join[1])
    sql_query += ';'
    print (sql_query)
    data = pd.read_sql_query(sql_query, con, index_col='index')

    # Standardize FIPS label
    data = data.rename (columns={'cnty_fips': 'FIPS'})
    data['FIPS'] =  data['FIPS'].apply (lambda x: str(x).zfill (5))

    # Make -1's nan's (CDC data not available values)
    data.where (data != -1, other=np.nan, inplace=True)
    return (data)


def get_cdc_data_vals_features (data, dropna=False):
    """
    Trim features that aren't needed for the model.

    :type   data: `pandas.DataFrame`
    :param  data: Dataframe of the CDC data

    :return: tuple of `pandas.DataFrame`s features and values
    """

    cols = data.columns.tolist ()
    cols_features = cols
    cols_features.remove ('Value')
    for col in CDC_SKIP_COLS:
        try:
            cols_features.remove (col)
        except:
            return

    if dropna:
        data_nona = data[['Value'] + cols_features].dropna ()
        return (data_nona[cols_features], data_nona['Value'])
    else:
        return (data[cols_features], data['Value'])


def normalize_features (features):
    mu = features.mean ()
    sigma = features.std ()
    features_norm = (features - mu) / sigma
    # mu = []
    # sigma = []
    # data_norm = data
    # for col in data.columns:
    #     mu_col = data.loc[:, col].mean ()
    #     sigma_col = data.loc[:, col].std ()
    #     data_norm.loc[:, col] = (data.loc[:, col] - mu_col) / sigma_col
    #     mu.append (mu_col)
    #     sigma.append (sigma_col)

    return (features_norm, mu, sigma)

def cost_function (model, X, y, lambda_reg=0.):
    """
    Squared error cost function with simple squared regularization.

    :type   model: `sklearn` model
    :param  model: Some trained scikit-learn model

    :type   X: `pandas.DataFrame`
    :param  X: A dataframe of the model features

    :type   y: `pandas.Series`
    :param  y: A series of the data results

    :type   lambda_reg: float
    :param  lambda_reg: Regularization parameter (default: 0)

    :return: Cost function value.
    """

    m = len (y)
    pred = model.predict (X)
    cost = 1. / (2. * m) * ((pred - y)**2).sum () + \
        lambda_reg / (2. * m) * (model.coef_**2).sum ()
    return (cost)


def cost_function_no_reg (model, X, y):
    """
    Squared error cost function.

    :type   model: `sklearn` model
    :param  model: Some trained scikit-learn model

    :type   X: `pandas.DataFrame`
    :param  X: A dataframe of the model features

    :type   y: `pandas.Series`
    :param  y: A series of the data results

    :return: Cost function value.
    """

    m = len (y)
    pred = model.predict (X)
    cost = 1. / (2. * m) * ((pred - y)**2).sum ()
    return (cost)


if __name__ == '__main__':

    dbname = 'stroke_prev'
    username = raw_input ('postgresql username: ')
    password = getpass.getpass ('password: ')

    con = get_db_connection (dbname, username, password)
    stroke_deaths_smoothed_indicators = get_cdc_data (
        con, 'cdc_stroke_deaths_all_smoothed_plus_indicators',
        join_tablenames=['cdc_all_heart_dis_deaths_all_smoothed',
                         'cdc_cor_heart_dis_deaths_all_smoothed',
                         'cdc_hypertension_deaths_all_smoothed',
                         'cdc_acute_myocard_infarc_deaths_all_smoothed'],
        join_ons=[('cnty_fips', 'cnty_fips'), ('cnty_fips', 'cnty_fips'),
                  ('cnty_fips', 'cnty_fips'), ('cnty_fips', 'cnty_fips'),
                  ],
        join_val_names=['all_heart_dis_death', 'cor_heart_dis_death',
                        'hyperten_death', 'acute_myocard_infarc_death']
    )

    # Drop counties with no target value
    stroke_deaths_smoothed_indicators = \
        stroke_deaths_smoothed_indicators.dropna (subset=['Value'])
    # Get features and target, dropping all counties missing data
    X_stroke_indicators_cdc, y_stroke_deaths = \
        get_cdc_data_vals_features (stroke_deaths_smoothed_indicators,
                                    dropna=False)

    # Impute missing data and rescale all explanatory features
    med_imputer = Imputer (strategy='median').fit (X_stroke_indicators_cdc)
    X_stroke_indicators_cdc_imp = med_imputer.transform (
        X_stroke_indicators_cdc)
    std_scale = StandardScaler ().fit (X_stroke_indicators_cdc_imp)
    X_stroke_indicators_cdc_imp_norm = std_scale.transform (
        X_stroke_indicators_cdc_imp)
    X_stroke_indicators_cdc_imp_norm = pd.DataFrame (
        X_stroke_indicators_cdc_imp_norm, index=X_stroke_indicators_cdc.index,
        columns=X_stroke_indicators_cdc.columns)

    # Get train, test data
    X_train, X_test, y_train, y_test = train_test_split (
        X_stroke_indicators_cdc_imp_norm, y_stroke_deaths,
        test_size=0.3,
        # random_state=100)
    )

    # Set model to use
    ml_linear = LinearRegression ()

    # Set cross-validation to use
    k_fold = KFold (len (y_train), n_folds=10)

    # Calculate cross-validation score
    cv_score = cross_val_score (ml_linear, X_train, y_train,
                                cv=k_fold, n_jobs=-1)
    print ('Cross-Validation R^2 score: {0:.4f} +/- {1:.4f}'.format (
        cv_score.mean (), cv_score.std ()))

    # Train the final model
    ml_linear.fit (X_train, y_train)
    print ('Test sample R^2 score: {0}'.format (
        ml_linear.score (X_test, y_test)))

    ml_dir = '/home/rmaunu/sandbox/insight/stroke_prev/ml'
    model_name = 'linear_model'

    print ('Saving model to `{0}/{1}.pickle`...'.format (ml_dir, model_name))
    ensure_dir (ml_dir)
    save (ml_linear, '{0}/{1}.pickle'.format (ml_dir, model_name))

