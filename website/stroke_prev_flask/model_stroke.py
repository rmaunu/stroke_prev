from __future__ import print_function

import sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import utils

INFLATION_RATE = 1.2226
prop_stroke = [0.78278, 0.21922]  # https://www.hcup-us.ahrq.gov/reports/statbriefs/sb51.pdf
cost_stroke = [9100, 19500]

cost_per_stroke = INFLATION_RATE * np.sum ([prop*cost for prop, cost
                                            in zip (prop_stroke, cost_stroke)])

demographic_features = ['perc_hisp', 'perc_white', 'perc_black', 'perc_aian',
                        'perc_api', 'perc_65up']

def get_stroke_pred (county_data):

    if len (county_data) != 1:
        return (('CHECK YOUR COUNTY INPUT', 'CHECK YOUR COUNTY INPUT'))

    model = utils.load ('stroke_prev_flask/ml/model.pickle')
    model_features_final = utils.load ('stroke_prev_flask/ml/model_features_final.pickle')
    model_features_imp = utils.load ('stroke_prev_flask/ml/model_features_imp.pickle')
    med_imputer = utils.load ('stroke_prev_flask/ml/med_imputer.pickle')
    std_scale = utils.load ('stroke_prev_flask/ml/std_scale.pickle')

    county_data = county_data[model_features_imp]
    county_data_ana = med_imputer.transform (county_data)
    county_data_ana = pd.DataFrame (
        county_data_ana, index=county_data.index,
        columns=county_data.columns)
    county_data_ana = std_scale.transform (county_data_ana)
    county_data_ana = pd.DataFrame (
        county_data_ana, index=county_data.index,
        columns=county_data.columns)
    county_data_ana = county_data_ana[model_features_final]
    print (county_data_ana)

    pred_hosp = model.predict (county_data_ana)[0]
    result = []
    result.append ('{0:.1f}'.format (pred_hosp))
    result.append ('${0:,d}'.format (
        int (pred_hosp / 1000. * county_data.iloc[0, :].loc['total_pop'] *
             county_data.iloc[0, :].loc['perc_65up'] / 100. * cost_per_stroke)
             / 10000 * 10000))

    return (result)

def get_stroke_pred_counties (county_data,
                              reduce_col={}):

    model = utils.load ('stroke_prev_flask/ml/model.pickle')
    model_features_final = utils.load ('stroke_prev_flask/ml/model_features_final.pickle')
    model_features_imp = utils.load ('stroke_prev_flask/ml/model_features_imp.pickle')
    med_imputer = utils.load ('stroke_prev_flask/ml/med_imputer.pickle')
    std_scale = utils.load ('stroke_prev_flask/ml/std_scale.pickle')

    county_data = county_data[model_features_imp]
    county_data_ana = med_imputer.transform (county_data)
    county_data_ana = pd.DataFrame (
        county_data_ana, index=county_data.index,
        columns=county_data.columns)
    for key in reduce_col.keys ():
        county_data_ana.loc[:, key] = county_data_ana.loc[:, key] * reduce_col[key]
    county_data_ana = std_scale.transform (county_data_ana)
    county_data_ana = pd.DataFrame (
        county_data_ana, index=county_data.index,
        columns=county_data.columns)
    county_data_ana = county_data_ana[model_features_final]

    pred_hosp = model.predict (county_data_ana)
    print (pred_hosp)
    return (pred_hosp)

def get_max_features (county_data, all_data, target_key='stroke_hosp'):

    if len (county_data) != 1:
        return (('CHECK YOUR COUNTY INPUT', 'CHECK YOUR COUNTY INPUT'))

    model = utils.load ('stroke_prev_flask/ml/model.pickle')
    model_features_final = utils.load ('stroke_prev_flask/ml/model_features_final.pickle')
    model_features_imp = utils.load ('stroke_prev_flask/ml/model_features_imp.pickle')
    med_imputer = utils.load ('stroke_prev_flask/ml/med_imputer.pickle')
    std_scale = utils.load ('stroke_prev_flask/ml/std_scale.pickle')

    y_all = all_data[target_key]
    all_data = all_data[model_features_imp]
    all_data_ana = med_imputer.transform (all_data)
    all_data_ana = pd.DataFrame (
        all_data_ana, index=all_data.index,
        columns=all_data.columns)
    all_data_ana = std_scale.transform (all_data_ana)
    all_data_ana = pd.DataFrame (
        all_data_ana, index=all_data.index,
        columns=all_data.columns)
    all_data_ana = all_data_ana[model_features_final]

    y_county = county_data[target_key].iloc[0]
    county_data = county_data[model_features_imp]
    county_data_ana = med_imputer.transform (county_data)
    county_data_ana = pd.DataFrame (
        county_data_ana, index=county_data.index,
        columns=county_data.columns)
    county_data_ana = std_scale.transform (county_data_ana)
    county_data_ana = pd.DataFrame (
        county_data_ana, index=county_data.index,
        columns=county_data.columns)
    county_data_ana = county_data_ana[model_features_final]

    imp_features = []
    imp_features.append ((utils.features_key[target_key],
                          '{0:.0f}'.format (float ((y_all > y_county).sum ()) / len (y_all) * 100)))

    xgb_imp = model.booster().get_score (importance_type='weight')  # XGBoost model
    total_splits = 0
    xgb_idx = []
    xgb_keys = []
    for key in xgb_imp.keys ():
        xgb_keys.append (key)
        total_splits += xgb_imp[key]

    xgb_vals = []
    for key in xgb_imp.keys ():
        xgb_vals.append (float (xgb_imp[key]) / total_splits)

    xgb_vals = np.array (xgb_vals)

    # for idx in np.abs (model.coef_).argsort ()[::-1][:10]:
    for idx in xgb_vals.argsort ()[::-1][:10]:
        key = xgb_keys[idx]
        x_county = county_data_ana.iloc[0, :][key]
        imp_features.append (
            (utils.features_key[key], '{0:.0f}'.format (float ((all_data_ana.loc[:, key] < x_county).sum ()) / len (y_all) * 100)))
    # try:
    #     for idx in np.abs (model.coef_).argsort ()[::-1][:10]:
    #         x_county = county_data_ana[0, idx]
    #         if model.coef_[idx] < 0:
    #             imp_features.append (
    #                 (utils.features_key[model_features_final[idx]], '{0:.0f}'.format (float ((all_data_ana[:, idx] < x_county).sum ()) / len (y_all) * 100)))
    #         else:
    #             imp_features.append (
    #                 (utils.features_key[model_features_final[idx]], '{0:.0f}'.format (float ((all_data_ana[:, idx] > x_county).sum ()) / len (y_all) * 100)))
    # except:
    #     for idx in model.feature_importances_.argsort ()[::-1][:10]:
    #         x_county = county_data_ana[0, idx]
    #         imp_features.append (
    #             (utils.features_key[model_features[idx]], '{0:.0f}'.format (float ((all_data_ana[:, idx] > x_county).sum ()) / len (y_all) * 100)))

    print (imp_features)
    return (imp_features)
