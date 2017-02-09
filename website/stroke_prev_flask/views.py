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
from collections import OrderedDict

from bokeh import layouts
from bokeh.embed import components
from bokeh.plotting import figure, output_file, ColumnDataSource
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from bokeh.palettes import Spectral4
from bokeh.models import HoverTool, OpenURL, TapTool, Jitter, Range1d, Spacer

from model_stroke import get_stroke_pred, get_stroke_pred_counties, \
    get_max_features, cost_per_stroke
from utils import strip_county_formatting, features_key, MODEL_RMSE

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


colors = {
    'Black': '#000000',
    'Red':   '#FF0000',
    'Green': '#00FF00',
    'Blue':  '#0000FF',
}

def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]


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

@app.route ('/county')
def county_data ():
    import us
    import addfips

    # pull 'stroke_month' from input field and store it
    county = request.args.get ('county')
    state = request.args.get ('state')
    variable = request.args.get ('variable')
    if variable is None:
        variable = 'htnadh_all'
    error = None

    if county == '' or state == '':
        error = 'Check your inputs'
        return render_template ("input.html", error=error)

    state_full = us.states.lookup (state).name
    fips_getter = addfips.AddFIPS ()

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

    plot_data = all_query_results.loc[:, ['FIPS', 'State', 'County',
                                          'total_pop', variable,
                                          'stroke_hosp']].dropna ()
    plot_data['FIPS_STATE'] = plot_data['State'].apply (fips_getter.get_state_fips)
    TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,tap,previewsave"
    if variable == 'log10_total_pop':
        fig = figure(plot_width=450, plot_height=450, x_axis_type="log",
                     tools=TOOLS, toolbar_location="above",
                     x_axis_location=None, y_axis_location=None,)
        fig.scatter (x={'field': 'total_pop', 'transform': Jitter(width=0.05)},
                     y={'field': 'stroke_hosp', 'transform': Jitter(width=0.05)},
                     source=ColumnDataSource (plot_data),
                     legend='US Counties', alpha=0.2, fill_color=Spectral4[0],
                     size=5,
                     line_color=Spectral4[0])
    else:
        fig = figure(plot_width=450, plot_height=450, tools=TOOLS,
                     toolbar_location="above", x_axis_location=None,
                     y_axis_location=None,)
        fig.scatter (x={'field': variable, 'transform': Jitter(width=0.05)},
                     y={'field': 'stroke_hosp', 'transform': Jitter(width=0.05)},
                     source=ColumnDataSource (plot_data),
                     legend='US Counties', alpha=0.2, fill_color=Spectral4[0],
                     size=5,
                     line_color=Spectral4[0])

    state_name = us.states.lookup (county[:2]).name
    if state_name != 'District of Columbia':
        idx_state = plot_data['State'] == state_name
        state_data = plot_data.loc[idx_state, :]
        if variable == 'log10_total_pop':
            fig.scatter (x={'field': 'total_pop', 'transform': Jitter(width=0.05)},
                         y={'field': 'stroke_hosp', 'transform': Jitter(width=0.05)},
                        source=ColumnDataSource (state_data),
                        legend='{0} Counties'.format (state_name),
                        size=7,
                        fill_color=Spectral4[2],
                        line_color=Spectral4[2])
        else:
            fig.scatter (x={'field': variable, 'transform': Jitter(width=0.05)},
                         y={'field': 'stroke_hosp', 'transform': Jitter(width=0.05)},
                         source=ColumnDataSource (state_data),
                         legend='{0} Counties'.format (state_name),
                         size=7,
                         fill_color=Spectral4[2],
                         line_color=Spectral4[2])

    idx_county = all_query_results['FIPS'] == county
    county_data = all_query_results.loc[idx_county, :].reset_index ()
    if (np.any (np.isnan (county_data['stroke_hosp']))):
        county_data.loc[:, 'stroke_hosp_pred'] = pd.Series ([float (get_stroke_pred (county_data)[0])])
        print (county_data['stroke_hosp_pred'])
        if variable == 'log10_total_pop':
            fig.scatter (x={'field': 'total_pop', 'transform': Jitter(width=0.05)},
                         y={'field': 'stroke_hosp_pred', 'transform': Jitter(width=0.05)},
                         source=ColumnDataSource (county_data),
                         size=10,
                         fill_color=Spectral4[-1], line_color=Spectral4[-1],
                         legend='{0}, {1}'.format (county_data['County'].tolist ()[0],
                                                   county_data['State'].tolist ()[0]))
        else:
            fig.scatter (x={'field': variable, 'transform': Jitter(width=0.05)},
                         y={'field': 'stroke_hosp_pred', 'transform': Jitter(width=0.05)},
                         source=ColumnDataSource (county_data),
                         size=10,
                         fill_color=Spectral4[-1], line_color=Spectral4[-1],
                         legend='{0}, {1}'.format (county_data['County'].tolist ()[0],
                                                   county_data['State'].tolist ()[0]))
    else:
        if variable == 'log10_total_pop':
            fig.scatter (x={'field': 'total_pop', 'transform': Jitter(width=0.05)},
                         y={'field': 'stroke_hosp', 'transform': Jitter(width=0.05)},
                         source=ColumnDataSource (county_data),
                         size=10,
                         fill_color=Spectral4[-1], line_color=Spectral4[-1],
                         legend='{0}, {1}'.format (county_data['County'].tolist ()[0],
                                                   county_data['State'].tolist ()[0]))
        else:
            fig.scatter (x={'field': variable, 'transform': Jitter(width=0.05)},
                         y={'field': 'stroke_hosp', 'transform': Jitter(width=0.05)},
                         source=ColumnDataSource (county_data),
                         size=10,
                         fill_color=Spectral4[-1], line_color=Spectral4[-1],
                         legend='{0}, {1}'.format (county_data['County'].tolist ()[0],
                                                   county_data['State'].tolist ()[0]))

    fig.xaxis.axis_label = features_key[variable]
    fig.yaxis.axis_label = features_key['stroke_hosp']


    hover = fig.select(dict(type=HoverTool))
    if variable == 'log10_total_pop':
        hover.tooltips = OrderedDict([
            ("(xx,yy)", "(@total_pop, @stroke_hosp)"),
            ("label", "@County" + ", " + "@State"),
        ])
    else:
        hover.tooltips = OrderedDict([
            ("(xx,yy)", "(@{0}, @stroke_hosp)".format (variable)),
            ("label", "@County" + ", " + "@State"),
        ])

    url = "/county?state=@FIPS_STATE&county=@FIPS&variable={0}".format (variable)
    taptool = fig.select(type=TapTool)
    taptool.callback = OpenURL(url=url)

    # Make histograms
    if variable == 'log10_total_pop':
        hhist, hedges = np.histogram(plot_data['total_pop'],
                                     bins=np.logspace(3., 7., 20))
        hzeros = np.zeros(len(hedges)-1)
        hmax = max(hhist)*1.1
        LINE_ARGS = dict(color="#3A5785", line_color=None)
    else:
        hhist, hedges = np.histogram(plot_data[variable], bins=20)
        hzeros = np.zeros(len(hedges)-1)
        hmax = max(hhist)*1.1
        LINE_ARGS = dict(color="#3A5785", line_color=None)

    if variable == 'log10_total_pop':
        ph = figure(toolbar_location=None, plot_width=fig.plot_width,
                    plot_height=200, x_range=fig.x_range, y_range=(0, hmax),
                    min_border=10, min_border_left=50, y_axis_location="left",
                    x_axis_type="log")
    else:
        ph = figure(toolbar_location=None, plot_width=fig.plot_width,
                    plot_height=200, x_range=fig.x_range, y_range=(0, hmax),
                    min_border=10, min_border_left=50, y_axis_location="left")
    ph.xgrid.grid_line_color = None
    # ph.yaxis.major_label_orientation = np.pi/4
    # ph.background_fill_color = "#fafafa"

    ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist,
            alpha=0.3, color=Spectral4[0], line_color="#3A5785")
    hh1 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros,
                  alpha=0.5, **LINE_ARGS)
    hh2 = ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros,
                  alpha=0.1, **LINE_ARGS)
    if variable == 'log10_total_pop':
        ph.ray(x=county_data['total_pop'], y=[0], length=1.e8, angle=np.pi/2,
            legend=False, line_color=Spectral4[-1], line_width=4)
    else:
        ph.ray(x=county_data[variable], y=[0], length=1000, angle=np.pi/2,
            legend=False, line_color=Spectral4[-1], line_width=4)

    ph.xaxis.axis_label = features_key[variable]
    ph.yaxis.axis_label = 'Count'

    vhist, vedges = np.histogram(plot_data['stroke_hosp'], bins=20)
    vzeros = np.zeros(len(vedges)-1)
    vmax = max(vhist)*1.1

    pv = figure(toolbar_location=None, plot_width=200,
                plot_height=fig.plot_height, x_range=(0, vmax),
                y_range=fig.y_range, min_border=10, y_axis_location="left")
    pv.ygrid.grid_line_color = None
    # pv.xaxis.major_label_orientation = np.pi/4

    pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist,
            alpha=0.3, color=Spectral4[0], line_color="#3A5785",
            )
    vh1 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
    vh2 = pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)
    pv.ray(x=[0], y=county_data['stroke_hosp'], length=1000, angle=0.,
           legend=False, line_color=Spectral4[-1], line_width=4)

    pv.yaxis.axis_label = features_key['stroke_hosp']
    pv.xaxis.axis_label = 'Count'

    layout = layouts.column(layouts.row(pv, fig), layouts.row(Spacer(width=200, height=200), ph))
    # layout = layouts.column(layouts.row(fig), layouts.row(ph))

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    # script, div = components(fig)
    script, div = components(layout)

    states = []
    counties = []
    for i, row in all_query_results.iterrows ():
        counties.append ([row['FIPS'][0:2], row['FIPS'], row['County']])
        if not states or states[-1][0] != row['FIPS'][0:2]:
            states.append ([row['FIPS'][0:2], row['State']])
        else:
            continue

    strokes = []
    # for i in range (10):
    for i in range (county_data.shape[0]):
        strokes.append (dict (county=county_data.iloc[i]['County'],
                              state=county_data.iloc[i]['State'],
                              stroke_hosp=county_data.iloc[i]['stroke_hosp'] if np.isfinite (county_data.iloc[i]['stroke_hosp']) else '---',
                              medicare_pop=int (county_data.iloc[i]['total_pop']*county_data.iloc[i]['perc_65up']/100.)/100*100
                              ))

    if len (county_data) == 1:
        pred_result = get_stroke_pred (county_data)
        imp_features = get_max_features (county_data, all_query_results)
    else:
        error = 'Check your inputs'
        return render_template ("input.html", error=error)

    return render_template ("county.html",
                            plot_script=script,
                            plot_div=div,
                            js_resources=js_resources,
                            css_resources=css_resources,
                            county_page=county,
                            state_page=state,
                            variable=variable,
                            counties=counties,
                            states=states,
                            strokes=strokes,
                            pred_result=pred_result,
                            pred_error='{0:.2f}'.format (MODEL_RMSE),
                            imp_features=imp_features,
                            error=error)

@app.route ('/intervention_result')
def intervention_result ():
    # pull 'stroke_month' from input field and store it

    variable = request.args.get ('variable')
    frac = 1. - float (request.args.get ('perc_red')) / 100.
    error = None

    query = """
        SELECT * FROM analysis_database_stroke;
        """

    query_results=pd.read_sql_query (query, con)

    if variable not in query_results.columns:
        error = 'Check your inputs'
        return ("")

    pred_result = get_stroke_pred_counties (query_results)
    pred_result_red = get_stroke_pred_counties (query_results, {variable: frac})
                                            # {'ltpia_prev_adj': 0.3})
                                            # {'airqual': 0.5})
                                            # {'daily_mean_smoking_2011': 0.5})
    stroke_red = pred_result - pred_result_red
    stroke_red[stroke_red < 0.] = 0

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

# @app.route ('/map_data')
# def stroke_data ():
#     # pull 'stroke_month' from input field and store it
#     query = """
#         SELECT * FROM analysis_database_stroke;
#         """

#     query_results=pd.read_sql_query (query, con)

#     strokes_data = []
#     # for i in range (10):
#     for i in range (query_results.shape[0]):
#         if int (query_results.iloc[i]['FIPS']) < 60000:
#             strokes_data.append ([
#                 query_results.iloc[i]['FIPS'], query_results.iloc[i]['County'],
#                 query_results.iloc[i]['State'],
#                 query_results.iloc[i]['stroke_hosp'] if np.isfinite (query_results.iloc[i]['stroke_hosp']) else '',
#                 query_results.iloc[i]['total_pop']*query_results.iloc[i]['perc_65up']/100.]
#             )

#     strokes_csv = [','.join (map (str, row)) for row in strokes_data]
#     header = "fips,county,state,stroke_hosp,medicare_pop\n"
#     output_csv = header + '\n'.join (strokes_csv)
#     # print (output_csv)
#     return (output_csv)

@app.route ('/reduce_map')
def stroke_red_map ():
    variable = request.args.get ('variable')
    frac = 1. - float (request.args.get ('perc_red')) / 100.
    error = None

    query = """
        SELECT * FROM analysis_database_stroke ORDER BY "State", "County";
        """

    query_results=pd.read_sql_query (query, con)
    query_results = query_results.loc[query_results['FIPS'].apply (int) < 60000, :]
    query_results['med_pop'] = query_results['total_pop'] * query_results['perc_65up'] / 100.

    if variable not in query_results.columns:
        error = 'Check your inputs'
        return ("")

    pred_result = get_stroke_pred_counties (query_results)
    pred_result_red = get_stroke_pred_counties (query_results, {variable: frac})
                                            # {'ltpia_prev_adj': 0.3})
                                            # {'airqual': 0.5})
                                            # {'daily_mean_smoking_2011': 0.5})
    stroke_red = pred_result - pred_result_red
    stroke_red[stroke_red < 0.] = 0
    total_reduce = np.array (stroke_red * query_results['med_pop'] / 1000.)
    idx_finite = np.isfinite (total_reduce)

    st_c_array = np.array (query_results[['State', 'County']])
    idx_sort = total_reduce.argsort ()[::-1]
    for i in idx_sort:
        print (total_reduce[i], st_c_array[i, :])

    total_reduce_err_sq = 2. * MODEL_RMSE**2 * total_reduce[idx_finite]**2
    total_reduce = total_reduce[idx_finite].sum ()
    total_reduce_err = np.sqrt (total_reduce_err_sq.sum ())
    print ('Total stroke reduction:', total_reduce, '+/-', total_reduce_err)
    cost_save = total_reduce * cost_per_stroke
    total_reduce = '{0:.0f}'.format (total_reduce)
    cost_save = '{0:,d}'.format (int (cost_save) / 10000 * 10000)

    strokes_data = []
    # for i in range (10):
    for i in range (query_results.shape[0]):
        strokes_data.append ([
            query_results.iloc[i]['FIPS'], query_results.iloc[i]['County'],
            query_results.iloc[i]['State'], stroke_red[i],
            query_results.iloc[i]['total_pop']*query_results.iloc[i]['perc_65up']/100.]
        )


    states = []
    counties = []
    for i, row in query_results.iterrows ():
        counties.append ([row['FIPS'][0:2], row['FIPS'], row['County']])
        if not states or states[-1][0] != row['FIPS'][0:2]:
            states.append ([row['FIPS'][0:2], row['State']])
        else:
            continue

    variable_txt = features_key[variable].lower ().replace (" (%)", "")
    perc_red = request.args.get ('perc_red')

    return render_template ("reduce_map.html",
                            states=states,
                            counties=counties,
                            variable=variable,
                            variable_txt=variable_txt,
                            total_reduce=total_reduce,
                            total_reduce_err='{0:.0f}'.format (total_reduce_err),
                            cost_save=cost_save,
                            perc_red=perc_red)

