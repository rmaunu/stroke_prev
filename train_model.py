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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import f_regression, RFECV
from xgboost.sklearn import XGBRegressor

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

from utils import save, load, resave, ensure_dir

sns.set_style ("white", rc=dict(font_scale=1.3))

# For CDC data, remove (not enough data):
#     FIPS, State, County,
#     demographic specific blood pressure med. adherence,
#     hospital number, specialist number
MODEL_FEATURES = [
    # 'dm_prev_adj',
    # 'ob_prev_adj',
    'ltpia_prev_adj',
    'no_hsdip',
    # 'no_college',
    # 'female_hd',
    'foodstmp',
    # 'home_val',
    'income',
    'GINI',
    # 'povpct',
    'unemploy',
    # 'perc_aian',
    # 'perc_api',
    # 'perc_black',
    # 'perc_white',
    # 'perc_hisp',
    'perc_65up',
    # 'total_pop',
    'log10_total_pop',
    'airqual',
    'PARKS',
    'sev_housing',
    'NCHS_collapse',
    'htnadh_all',
    # 'htnadh_white',
    # 'htnadh_black',
    # 'htnadh_hisp',
    # 'htnadh_api',
    # 'htnadh_aian',
    # 'diuradh',
    # 'rasadh',
    # 'stg_hosp',
    # 'stg_int_cr',
    # 'stg_rehab',
    # 'stg_emerg',
    # 'stg_neuro',
    'pharmpc',
    'prim_dr',
    # 'cvd_dr',
    # 'neuro_dr',
    # 'surg_dr',
    'pctui',
    # 'medicaid',
    # 'stroke_hosp',  # Target
    # 'stroke_death',
    # 'all_heart_dis_hosp',
    # 'card_dysrhythm_hosp',
    # 'cor_heart_dis_hosp',
    # 'hyperten_hosp',
    # 'acute_myocard_infarc_hosp',
    # 'heart_hosp',
    # 'heart_death',
    # 'any_alcohol_2011',
    'heavy_alcohol_2011',
    # 'binge_alcohol_2011',
    'daily_mean_smoking_2011',
    # 'total_mean_smoking_2011',
    'snap_redemp_per_store_2012',
    # 'pct_diabetes_adults_2010',
    'pct_obese_adults_2013',
    'rec_fac_2012',
    'ers_nat_amenity_index_1999',
    'food_insec_house_pct_10_12',
    'low_access_food_pct10',
    'low_access_grocery_pct10',
    # 'low_access_food_low_inc_pct10',
    # 'low_access_food_snr_pct10',
    # 'low_access_food_no_car_pct10',
    # 'very_low_food_insec_house_pct_10_12',
]

CDC_SKIP_COLS = ['FIPS', 'County', 'State',  # Labels
                 'perc_white', 'perc_black', 'perc_api', 'perc_aian', 'perc_hisp',  # Demographics
                 'htnadh_white', 'htnadh_black', 'htnadh_api', 'htnadh_aian', 'htnadh_hisp',  # Not enough adherence data
                 'stg_hosp', 'stg_int_cr', 'stg_rehab', 'stg_emerg', 'stg_neuro',  # Not enough clinic data
                 'cvd_dr', 'neuro_dr', 'surg_dr',
                 'stroke_death', 'all_heart_dis_death', 'cor_heart_dis_death',  # Skip correlated heart diseases for now
                 'hyperten_death', 'acute_myocard_infarc_death',
                 'stroke_hosp', 'all_heart_dis_hosp', 'card_dysrhythm_hosp'
                 'cor_heart_dis_hosp', 'hyperten_hosp',
                 'acute_myocard_infarc_hosp',
                 ]


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


def get_cdc_data (connection, primary_tablename, primary_val_name,
                  join_tablenames=[],
                  join_ons=[],
                  join_val_names=[]):
    """
    Get CDC data from the SQL database.

    :type   connection: database connection (e.g. `psycopg2`)
    :param  connection: A connection to an SQL database.

    :type   primary_tablename: str
    :param  primary_tablename: Name of the table to open.

    :type   primary_val_name: str
    :param  primary_val_name: Name to rename `Value` in the primary table to.

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
    # print (sql_query)
    data = pd.read_sql_query(sql_query, con, index_col='index')

    # Standardize FIPS label
    data = data.rename (columns={
        'cnty_fips': 'FIPS', 'Value': primary_val_name
    })
    data['FIPS'] =  data['FIPS'].apply (lambda x: str(x).zfill (5))

    # Make -1's nan's (CDC data not available values)
    data.where (data != -1, other=np.nan, inplace=True)
    return (data)


def get_ihme_alcohol_data (connection):
    """
    Get IHME data from the SQL database.

    :type   connection: database connection (e.g. `psycopg2`)
    :param  connection: A connection to an SQL database.

    :return: `pandas.DataFrame` of the data
    """

    alcohol_tables = ['any', 'heavy', 'binge']

    # build query
    for table in alcohol_tables:
        sql_query = """
SELECT "index", "FIPS", "State", "County", "2011 Both Sexes" AS "{0}_alcohol_2011"
    FROM ihme_alcohol_use_{0}_2002_2012_y2015m04d23;
        """.format (table)
        # print (sql_query)
        if table == alcohol_tables[0]:
            data = pd.read_sql_query(sql_query, con, index_col='index')
            data.where ((pd.notnull (data)), other=np.nan, inplace=True)
            data = data.dropna (subset=['FIPS'])
            data['FIPS'] =  data['FIPS'].apply (lambda x: str(x).zfill (5))
        else:
            data_tmp = pd.read_sql_query(sql_query, con, index_col='index')
            data_tmp.where ((pd.notnull (data_tmp)), other=np.nan, inplace=True)
            data_tmp['FIPS'] =  data_tmp['FIPS'].apply (lambda x: str(x).zfill (5))
            data = pd.merge (data,
                             data_tmp[['FIPS', '{0}_alcohol_2011'.format (table)]],
                             on="FIPS", how="left")

    # Make -1's nan's (CDC data not available values)
    return (data)


def get_ihme_smoking_data (connection):
    """
    Get IHME data from the SQL database.

    :type   connection: database connection (e.g. `psycopg2`)
    :param  connection: A connection to an SQL database.

    :return: `pandas.DataFrame` of the data
    """

    sql_query = """
SELECT "index", "FIPS", "state", "county", "year", "sex", "total_mean" AS "total_mean_smoking_2011",
"daily_mean" AS "daily_mean_smoking_2011"
FROM ihme_us_county_total_and_daily_smoking_prevalence_1996_2012;
    """
    # print (sql_query)

    data = pd.read_sql_query(sql_query, con, index_col='index')
    data = data.loc[data['year'] == 2011, :]
    data = data.loc[data['sex'] == 'Both', :]
    data.where ((pd.notnull (data)), other=np.nan, inplace=True)
    data = data.dropna (subset=['FIPS'])
    data['FIPS'] =  data['FIPS'].apply (lambda x: str(x).zfill (5))

    # Make -1's nan's (CDC data not available values)
    return (data)


def get_usda_food_data (connection):
    """
    Get IHME data from the SQL database.

    :type   connection: database connection (e.g. `psycopg2`)
    :param  connection: A connection to an SQL database.

    :return: `pandas.DataFrame` of the data
    """

    tables = ['usda_food_access_feb2014', 'usda_food_assistance_feb2014',
              'usda_food_health_feb2014', 'usda_food_insecurity_feb2014',
              'usda_food_stores_feb2014']

    for table in tables:
        if table == tables[0]:
            sql_query = """
SELECT "FIPS",
"PCT_LACCESS_POP10" AS "low_access_food_pct10",
"PCT_LACCESS_LOWI10" AS "low_access_food_low_inc_pct10",
"PCT_LACCESS_SENIORS10" AS "low_access_food_snr_pct10",
"PCT_LACCESS_HHNV10" AS "low_access_food_no_car_pct10"
FROM {0};
    """.format (table)
            # print (sql_query)
        elif table == tables[1]:
            sql_query = """
SELECT "FIPS",
"REDEMP_SNAPS12" AS "snap_redemp_per_store_2012"
FROM {0};
    """.format (table)
            # print (sql_query)
        elif table == tables[2]:
            sql_query = """
SELECT "FIPS",
"PCT_DIABETES_ADULTS10" AS "pct_diabetes_adults_2010",
"PCT_OBESE_ADULTS13" AS "pct_obese_adults_2013",
"RECFACPTH12" AS "rec_fac_2012",
"NATAMEN" AS "ers_nat_amenity_index_1999"
FROM {0};
    """.format (table)
            # print (sql_query)
        elif table == tables[3]:
            sql_query = """
SELECT "FIPS",
"FOODINSEC_10_12" AS "food_insec_house_pct_10_12",
"VLFOODSEC_10_12" AS "very_low_food_insec_house_pct_10_12"
FROM {0};
    """.format (table)
            # print (sql_query)
        elif table == tables[4]:
            sql_query = """
SELECT "FIPS",
"GROCPTH12" AS "low_access_grocery_pct10"
FROM {0};
    """.format (table)
            # print (sql_query)

        if table == tables[0]:
            data = pd.read_sql_query(sql_query, con)
            data.where ((pd.notnull (data)), other=np.nan, inplace=True)
            data = data.dropna (subset=['FIPS'])
            data['FIPS'] =  data['FIPS'].apply (lambda x: str(x).zfill (5))
        else:
            data_tmp = pd.read_sql_query(sql_query, con)
            data_tmp.where ((pd.notnull (data_tmp)), other=np.nan, inplace=True)
            data_tmp = data_tmp.dropna (subset=['FIPS'])
            data_tmp['FIPS'] =  data_tmp['FIPS'].apply (lambda x: str(x).zfill (5))
            data = pd.merge (data, data_tmp, on="FIPS", how="left")

    return (data)


def get_cdc_data_vals_features (data, target_key, dropna=False):
    """
    Trim features that aren't needed for the model.

    :type   data: `pandas.DataFrame`
    :param  data: Dataframe of the CDC data

    :return: tuple of `pandas.DataFrame`s features and values
    """

    if dropna:
        data_nona = data[[target_key] + MODEL_FEATURES].dropna ()
        return (data_nona[MODEL_FEATURES], data_nona[target_key])
    else:
        return (data[MODEL_FEATURES], data[target_key])


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
    cdc_data = get_cdc_data (
        con, 'cdc_stroke_deaths_all_plus_indicators', 'stroke_death',
        join_tablenames=['cdc_stroke_hosp_65plus',
                         'cdc_all_heart_dis_hosp_65plus',
                         'cdc_card_dysrhyth_hosp_65plus',
                         'cdc_cor_heart_dis_hosp_65plus',
                         'cdc_hypertension_hosp_65plus',
                         'cdc_acute_myocard_infarc_hosp_65plus',
                         'cdc_all_heart_dis_deaths_all'
                         ],
        join_ons=[('cnty_fips', 'cnty_fips'), ('cnty_fips', 'cnty_fips'),
                  ('cnty_fips', 'cnty_fips'), ('cnty_fips', 'cnty_fips'),
                  ('cnty_fips', 'cnty_fips'), ('cnty_fips', 'cnty_fips'),
                  ('cnty_fips', 'cnty_fips')],
        join_val_names=['stroke_hosp', 'all_heart_dis_hosp',
                        'card_dysrhythm_hosp',
                        'cor_heart_dis_hosp', 'hyperten_hosp',
                        'acute_myocard_infarc_hosp', 'all_heart_dis_death']
    )

    alcohol_data = get_ihme_alcohol_data (con)
    smoking_data = get_ihme_smoking_data (con)
    food_data = get_usda_food_data (con)

    all_data = pd.merge (cdc_data,
                         alcohol_data.loc[:, ['FIPS', 'any_alcohol_2011',
                                              'heavy_alcohol_2011',
                                              'binge_alcohol_2011']],
                         on='FIPS', how='left')

    all_data = pd.merge (all_data,
                            smoking_data.loc[:, ['FIPS', 'total_mean_smoking_2011',
                                                 'daily_mean_smoking_2011']],
                            on='FIPS', how='left')

    all_data = pd.merge (all_data, food_data, on='FIPS', how='left')

    # Dump table to SQL
    engine = create_engine ('postgres://{0}:{1}@localhost/{2}'.format (
        username, password, dbname),
        pool_recycle=7200, pool_size=10)


    # Drop counties with no target value
    # stroke_data = stroke_data.dropna (subset=['Value'])
    all_data['heart_hosp'] = all_data[['stroke_hosp', 'all_heart_dis_hosp']].sum (axis=1)
    all_data['heart_death'] = all_data[['stroke_death', 'all_heart_dis_death']].sum (axis=1)
    # target_key = 'heart_hosp'
    # target_key = 'heart_death'
    target_key = 'stroke_hosp'
    all_data['log10_total_pop'] = np.log10 (all_data['total_pop'])
    all_data.to_sql ('analysis_database_stroke', engine,
                        if_exists='replace')

    all_data = all_data.dropna (subset=[target_key])

    # Get features and target, dropping all counties missing data
    X_stroke_indicators_cdc, y_stroke = \
        get_cdc_data_vals_features (all_data, target_key, dropna=True)
        # get_cdc_data_vals_features (all_data, target_key, dropna=False)
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

    plt_dir = ensure_dir ('/home/rmaunu/sandbox/insight/stroke_prev/plots')
    ml_dir = ensure_dir ('/home/rmaunu/sandbox/insight/stroke_prev/ml')

    # Get train, test data
    X_train, X_test, y_train, y_test = train_test_split (
        X_stroke_indicators_cdc_imp_norm, y_stroke,
        test_size=0.3, random_state=0)

    print ('Training size:', len (X_train))
    print ('Testing size:', len (X_test))

    save (X_train.columns, '{0}/model_features_imp.pickle'.format (ml_dir))

    # # Linear regression p-value feature selection
    # F_vals, p_vals = f_regression (X_train, y_train, center=True)
    # print ('Dropping features with p > 0.05:', X_train.columns[p_vals >= 0.05])
    # X_train_selection = X_train.loc[:, p_vals < 0.05]
    # X_test_selection = X_test.loc[:, p_vals < 0.05]
    X_train_selection = X_train
    X_test_selection = X_test

    save (X_train_selection, '{0}/X_train.pickle'.format (ml_dir))
    save (X_test_selection, '{0}/X_test.pickle'.format (ml_dir))
    save (y_train, '{0}/y_train.pickle'.format (ml_dir))
    save (y_test, '{0}/y_test.pickle'.format (ml_dir))


    # Set cross-validation to use
    k_fold = KFold (len (y_train), n_folds=5, shuffle=True)

    # Set model to use

    # model_name = 'linear_model'
    # model_name = 'linear_lasso_model'
    # model_name = 'linear_ridge_model'
    # model_name = 'elasticnet_model'
    model_name = 'random_forest_model'
    # model_name = 'xgboost_forest_model'
    # model_name = 'svr_model'

    if model_name == 'linear_model':
        ml_model = LinearRegression ()
    elif model_name == 'linear_lasso_model':
        ml_model = Lasso ()  # alpha=0.1, R^2=0.41
        param_grid = {'alpha': np.logspace (-3, 3, 13)}
        grid_search_k_fold = GridSearchCV (ml_model, param_grid, cv=k_fold,
                                           verbose=3, n_jobs=-1)
    elif model_name == 'linear_ridge_model':
        ml_model = Ridge (alpha=100)  # alpha=100, R^2=0.41
        param_grid = {'alpha': np.logspace (-3, 3, 13)}
        grid_search_k_fold = GridSearchCV (ml_model, param_grid, cv=k_fold,
                                           verbose=3, n_jobs=-1)
    elif model_name == 'elasticnet_model':
        ml_model = ElasticNet ()
        param_grid = {'alpha': np.logspace (-3, 3, 13),
                      'l1_ratio': np.linspace (0, 1, 10)}
        grid_search_k_fold = GridSearchCV (ml_model, param_grid, cv=k_fold,
                                        verbose=3, n_jobs=-1)
    elif model_name == 'random_forest_model':
        ml_model = RandomForestRegressor (n_jobs=-1)  # 'n_estimators': 50, 'max_features': 'auto', 'n_jobs': -1, 'max_depth': 6, 'min_samples_leaf': 5, R^2=0.50
        param_grid = {'max_depth': [5, 6, 7, 8], 'n_estimators': [50, 75, 100, 200],
                      'min_samples_leaf': [2, 5, 10], 'n_jobs': [-1],
                      'max_features': ['auto', 'sqrt', 'log2']}
        grid_search_k_fold = RandomizedSearchCV (ml_model, param_grid, cv=k_fold,
                                                 n_iter=100, verbose=3, n_jobs=-1)
    elif model_name == 'xgboost_forest_model':
        ml_model = XGBRegressor (
            learning_rate=0.1,
            n_estimators=140,
            max_depth=4,
            min_child_weight=2,
            gamma=1.e-3,
            subsample=0.85,
            colsample_bytree=0.55,
            reg_alpha=1,
            nthread=4,
            scale_pos_weight=1)
        param_grid = {
            # 'max_depth': range (3, 10, 2),
            # 'min_child_weight': range (1, 6, 2),
            # 'max_depth': [2, 3, 4],
            # 'min_child_weight': [2, 3, 4],
            # 'min_child_weight':[6, 8, 10, 12],
            # 'gamma': np.linspace (0, 0.4, 5),
            # 'subsample': np.linspace (0.6, 0.9, 4),
            # 'colsample_bytree': np.linspace (0.6, 0.9, 4),
            # 'subsample': np.linspace (0.8, 0.95, 4),
            # 'colsample_bytree': np.linspace (0.55, 0.7, 4),
            # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
            # 'reg_alpha':[0, 1e-5, 1.e-4, 1.e-3, 1e-2],
            'learning_rate': [0.01],
            'n_estimators': [1000],
        }
        grid_search_k_fold = GridSearchCV (ml_model, param_grid, cv=k_fold,
                                           verbose=3, n_jobs=-1)
        # ml_model.set_params (**grid_search_k_fold.best_params_)

    elif model_name == 'svr_model':
        ml_model = SVR ()
        param_grid = {'C': np.logspace (-1, 3, 13), 'epsilon': [0.1],
                      'kernel': ['rbf', 'linear']}
        grid_search_k_fold = GridSearchCV (ml_model, param_grid, cv=k_fold,
                                        verbose=3, n_jobs=-1)

    # # Feature selection
    # selector = RFECV (ml_model, cv=k_fold)
    # selector.fit (X_train_selection, y_train)
    # save (selector, '{0}/ridge_rfe_selector.pickle'.format (ml_dir))

    # # X_train_selection = X_train
    # # X_test_selection = X_test
    # print ('Selected features:', X_train_selection.columns[selector.support_])
    # X_train_selection = X_train_selection.loc[:, selector.support_]
    # X_test_selection = X_test_selection.loc[:, selector.support_]
    # save (X_train_selection, '{0}/X_train_selection.pickle'.format (ml_dir))

    # Plot correlation matrix of final variables
    corr = X_train_selection.corr ()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig = plt.figure (figsize=(16, 16))
    ax = fig.add_subplot (111)
    sns.heatmap(corr,  mask=mask,  vmax=1,  square=True, linewidths=.5,  cbar_kws={"shrink": .5}, ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.savefig ('{0}/corr_heatmap_selection_{1}.png'.format (plt_dir, model_name), transparent=False)

    if model_name == 'linear_model':
        cv_model = cross_val_score (ml_model, X_train_selection, y_train, cv=k_fold)
        save (cv_model, '{0}/cv_{1}.pickle'.format (ml_dir, model_name))
        print ('CV R^2 score: {0} +/- {1}'.format (cv_model.mean (), cv_model.std ()))

        ml_model.fit (X_train_selection, y_train)
        print ('Test sample R^2 score: {0}'.format (
            ml_model.score (X_test_selection, y_test)))
    else:
        grid_search_k_fold.fit (X_train_selection, y_train)
        save (grid_search_k_fold, '{0}/grid_search_{1}.pickle'.format (ml_dir, model_name))

        # Train the final model
        print ('Best grid search score:', grid_search_k_fold.best_score_)
        ml_model.set_params (**grid_search_k_fold.best_params_)
        cv_model = cross_val_score (ml_model, X_train_selection, y_train, cv=k_fold)
        print ('CV R^2 score: {0} +/- {1}'.format (cv_model.mean (), cv_model.std ()))

        print ('Fitting with best model parameters:', grid_search_k_fold.best_params_)
        ml_model.fit (X_train_selection, y_train)
        print ('Test sample R^2 score: {0}'.format (
            ml_model.score (X_test_selection, y_test)))

    fig = plt.figure ()
    ax = fig.add_subplot (111)
    # ax.scatter (y_train, ml_model.predict (X_train_selection),
                # marker='.', color='b', edgecolor='b', alpha=0.2, s=40,
                # label='Train')
    ax.scatter (y_test, ml_model.predict (X_test_selection),
                marker='.', color='r', edgecolor='r', alpha=0.5, s=40,
                label='Test')
    ax.plot (np.linspace (0, 1000, 2), np.linspace (0, 1000, 2),
             ls='--', color='k')
    ax.grid ()
    # ax.legend (loc='lower right')
    ax.set_xlabel ('Actual Stroke Hospitalization Rate (per 1000 Medicare Ben.)')
    ax.set_ylabel ('Predicted Stroke Hospitalization Rate (per 1000 Medicare Ben.)')
    ax.set_xlim (3, 23)
    ax.set_ylim (3, 23)
    ax.set_aspect ('equal')
    fig.savefig ('{0}/{1}_prediction.png'.format (plt_dir, model_name),
                 bbox_inches='tight')

    print ('Saving model to `{0}/{1}.pickle`...'.format (ml_dir, model_name))
    ensure_dir (ml_dir)
    save (X_train_selection.columns, '{0}/model_features_final.pickle'.format (ml_dir))
    save (ml_model, '{0}/{1}.pickle'.format (ml_dir, model_name))
    save (med_imputer, '{0}/med_imputer.pickle'.format (ml_dir))
    save (std_scale, '{0}/std_scale.pickle'.format (ml_dir))

