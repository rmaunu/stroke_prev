from __future__ import print_function

import us
import psycopg2
import pandas as pd
import numpy as np

from itertools import izip
from flask import render_template
from flask import request
from stroke_prev_flask import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

from model_stroke import get_stroke_pred, get_stroke_pred_counties, \
    get_max_features
from utils import strip_county_formatting

# user = 'rmaunu'
user = 'postgres'
password = 'scruzslugs97'
host = 'localhost'
dbname = 'stroke_prev'

db = create_engine ('postgres://{0}:{1}@{2}/{3}'.format (
    user, password, host, dbname))
con = None
con = psycopg2.connect (database=dbname,
                        user=user,
                        password=password,
                        host='/var/run/postgresql')

@app.route ('/db_fancy')
def strokes_page_fancy ():
    # sql_query = \
    #     """
    #     SELECT "County", "State", total_pop, perc_65up, stroke_hosp FROM
    #         analysis_database_stroke;
    #     """

    sql_query = \
        """
        SELECT * FROM analysis_database_stroke;
        """

    query_results = pd.read_sql_query (sql_query, con)
    strokes = []
    for i in range (10):
        strokes.append (dict (county=query_results.iloc[i]['County'],
                              state=query_results.iloc[i]['State'],
                              stroke_hosp=query_results.iloc[i]['stroke_hosp'],
                              medicare_pop=int (query_results.iloc[i]['total_pop']*query_results.iloc[i]['perc_65up']/100.)/100*100
                              ))
    return render_template ('strokes.html', strokes=strokes)

@app.route ('/')
def landing_page ():
    return render_template ("cover.html")

@app.route ('/about')
def about_page ():
    return render_template ("about.html")

@app.route ('/contact')
def contact_page ():
    return render_template ("contact.html")

@app.route ('/input')
def stroke_input ():
    query = """
        SELECT "FIPS", "County", "State" FROM analysis_database_stroke ORDER BY "State", "County";
        """

    all_query_results = pd.read_sql_query (query, con)
    all_query_results = all_query_results.loc[all_query_results['FIPS'].apply (int) < 60000, :]

    states = []
    counties = []
    for i, row in all_query_results.iterrows ():
        counties.append ([row['FIPS'][0:2], row['FIPS'], row['County']])
        if not states or states[-1][0] != row['FIPS'][0:2]:
            states.append ([row['FIPS'][0:2], row['State']])
        else:
            continue

    print (counties)
    return render_template ("input.html",
                            counties=counties,
                            states=states,
                            )

@app.route ('/output')
def strokes_output ():
    # pull 'stroke_month' from input field and store it
    county = request.args.get ('county')
    state = request.args.get ('state')
    error = None

    print (county, state)
    if county == '' or state == '':
        error = 'Check your inputs'
        return render_template ("input.html", error=error)

    state_full = us.states.lookup (state).name

    # Grab everything, it's not too big
    # query = """
    #     SELECT "County", "State", total_pop, perc_65up, stroke_hosp FROM
    #         analysis_database_stroke;
    #     """

    query = """
        SELECT * FROM analysis_database_stroke ORDER BY "State", "County";
        """

    all_query_results=pd.read_sql_query (query, con)
    all_query_results = all_query_results.loc[all_query_results['FIPS'].apply (int) < 60000, :]

    states = []
    counties = []
    for i, row in all_query_results.iterrows ():
        counties.append ([row['FIPS'][0:2], row['FIPS'], row['County']])
        if not states or states[-1][0] != row['FIPS'][0:2]:
            states.append ([row['FIPS'][0:2], row['State']])
        else:
            continue

    county_row = np.array ([
        county == county_query for county_query in all_query_results['FIPS']])
    query_results=all_query_results.loc[county_row]
    print (query_results)

    strokes = []
    # for i in range (10):
    for i in range (query_results.shape[0]):
        strokes.append (dict (county=query_results.iloc[i]['County'],
                              state=query_results.iloc[i]['State'],
                              stroke_hosp=query_results.iloc[i]['stroke_hosp'] if np.isfinite (query_results.iloc[i]['stroke_hosp']) else '---',
                              medicare_pop=int (query_results.iloc[i]['total_pop']*query_results.iloc[i]['perc_65up']/100.)/100*100
                              ))
    print (strokes)

    if len (query_results) == 1:
        pred_result = get_stroke_pred (query_results)
        imp_features = get_max_features (query_results, all_query_results)
    else:
        error = 'Check your inputs'
        return render_template ("input.html", error=error)

    return render_template ("output.html",
                            counties=counties,
                            states=states,
                            strokes=strokes,
                            pred_result=pred_result,
                            imp_features=imp_features,
                            error=error)

@app.route ('/intervention_result')
def intervention_result ():
    # pull 'stroke_month' from input field and store it

    var = request.args.get ('var')
    frac = float (request.args.get ('frac'))
    error = None

    query = """
        SELECT * FROM analysis_database_stroke;
        """

    query_results=pd.read_sql_query (query, con)

    if var not in query_results.columns:
        error = 'Check your inputs'
        return ("")

    pred_result = get_stroke_pred_counties (query_results)
    pred_result_red = get_stroke_pred_counties (query_results, {var: frac})
                                            # {'ltpia_prev_adj': 0.3})
                                            # {'airqual': 0.5})
                                            # {'daily_mean_smoking_2011': 0.5})
    stroke_red = pred_result - pred_result_red

    strokes_data = []
    # for i in range (10):
    for i in range (query_results.shape[0]):
        if int (query_results.iloc[i]['FIPS']) < 60000:
            strokes_data.append ([
                query_results.iloc[i]['FIPS'], query_results.iloc[i]['County'],
                query_results.iloc[i]['State'], stroke_red[i],
                query_results.iloc[i]['total_pop']*query_results.iloc[i]['perc_65up']/100.]
            )

    strokes_csv = [','.join (map (str, row)) for row in strokes_data]
    header = "fips,county,state,stroke_red,medicare_pop\n"
    output_csv = header + '\n'.join (strokes_csv)
    # print (output_csv)
    return (output_csv)

@app.route ('/map_data')
def stroke_data ():
    # pull 'stroke_month' from input field and store it
    query = """
        SELECT * FROM analysis_database_stroke;
        """

    query_results=pd.read_sql_query (query, con)

    strokes_data = []
    # for i in range (10):
    for i in range (query_results.shape[0]):
        if int (query_results.iloc[i]['FIPS']) < 60000:
            strokes_data.append ([
                query_results.iloc[i]['FIPS'], query_results.iloc[i]['County'],
                query_results.iloc[i]['State'],
                query_results.iloc[i]['stroke_hosp'] if np.isfinite (query_results.iloc[i]['stroke_hosp']) else '',
                query_results.iloc[i]['total_pop']*query_results.iloc[i]['perc_65up']/100.]
            )

    strokes_csv = [','.join (map (str, row)) for row in strokes_data]
    header = "fips,county,state,stroke_hosp,medicare_pop\n"
    output_csv = header + '\n'.join (strokes_csv)
    # print (output_csv)
    return (output_csv)

@app.route ('/reduce_map')
def stroke_red_map ():
    var = request.args.get ('var')
    frac = float (request.args.get ('frac'))
    return render_template ("red_map.html", var=var, frac=frac)

@app.route ('/map')
def stroke_map ():
    return render_template ("map.html")
