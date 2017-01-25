#!/usr/bin/env python

from __future__ import print_function

import getpass
import re
import us
import psycopg2
import progressbar
import addfips
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

def parse_pcen_race_gen (race_gen):
    race_gen_dict = {
        '1': ('White', 'Male'),
        '2': ('White', 'Female'),
        '3': ('Black or African American', 'Male'),
        '4': ('Black or African American', 'Female'),
        '5': ('American Indian or Alaska Native', 'Male'),
        '6': ('American Indian or Alaska Native', 'Female'),
        '7': ('Asian or Pacific Islander', 'Male'),
        '8': ('Asian or Pacific Islander', 'Female'),
    }

    try:
        return (race_gen_dict[race_gen])
    except:
        return (('NAN', 'NAN'))

def parse_pcen (line):
    series = line[:4]
    fips = line[4:9]
    age = int (line[9:11])
    race, gen = parse_pcen_race_gen (line[11])
    hisp_origin = int (line[12]) - 0
    pop_est = line[13:].split ()

    return ([series, fips, age, race, gen, hisp_origin] + pop_est)

def make_table (filename, tablename, engine,
                if_exists='append'):

    if not database_exists (engine.url):
        create_database (engine.url)

    if 'diabetes' in filename:
        data = pd.read_csv (filename, header=1)
    elif 'pcen' in filename:
        data = pd.DataFrame (columns=['Series', 'FIPS', 'Age', 'Race', 'Gender',
                                      'Hispanic Origin', 'POP2010_apr',
                                      'POP2010_jul', 'POP2011', 'POP2012',
                                      'POP2013', 'POP2014', 'POP2015'])

        num_lines = sum (1 for line in open (filename))
        split = 500000

        widgets = [progressbar.Percentage (), progressbar.Bar (),
                   progressbar.FormatLabel (
                       'Processed: %(value)d lines (in: %(elapsed)s)')]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=num_lines)
        data_list = []
        with open (filename) as f:
            for i, line in pbar (enumerate (f)):
                line_vals = parse_pcen (line)
                data_list.append (line_vals)

                if i % split == 0:
                    data = pd.DataFrame (
                        data_list, columns=['Series', 'FIPS', 'Age', 'Race',
                                            'Gender', 'Hispanic Origin',
                                            'POP2010_apr', 'POP2010_jul',
                                            'POP2011', 'POP2012', 'POP2013',
                                            'POP2014', 'POP2015'])

                    data.index = data.index + split * int (i / split - 1)

                    if i == split:
                        data.to_sql (tablename, engine, if_exists=if_exists)
                    else:
                        data.to_sql (tablename, engine, if_exists='append')

                    data_list = []

            if num_lines % split != 0:
                data = pd.DataFrame (
                    data_list, columns=['Series', 'FIPS', 'Age', 'Race',
                                        'Gender', 'Hispanic Origin',
                                        'POP2010_apr', 'POP2010_jul',
                                        'POP2011', 'POP2012', 'POP2013',
                                        'POP2014', 'POP2015'])
                data.index = data.index + split * int (num_lines / split)
                data.to_sql (tablename, engine, if_exists='append')

        return

        # data[['Age', 'Hispanic Origin', 'POP2010_apr', 'POP2010_jul', 'POP2011',
        #       'POP2012', 'POP2013', 'POP2014', 'POP2015']] = \
        #     data[['Age', 'Hispanic Origin', 'POP2010_apr', 'POP2010_jul',
        #         'POP2011', 'POP2012', 'POP2013', 'POP2014', 'POP2015']].astype (int)

    else:
        data = pd.read_csv (filename)

    # file specific formatting
    if 'OBESITY' in filename:
        colnames = data.columns
        colnames = [re.sub ('Prevalence\ ', '', name) for name in colnames]
        colnames = [re.sub ('\ \(\%\)', '', name) for name in colnames]
        data.columns = colnames
        data['County'] = data['County'].apply (lambda x: x.split (',')[0] + ' County')
    elif 'HYPERTENSION' in filename:
        def p2f (val):
            return (float (val.strip ('%')))

        for col in data.columns[4:]:
            data[col] = data[col].apply (p2f).astype (float)

        data = pd.melt (data, id_vars=data.columns.tolist ()[:4],
                        value_vars=data.columns.tolist ()[4:],
                        var_name=u'sample', value_name=u'percent')
        data['Sample Type'] = data['sample'].apply (lambda x: x.split (',')[0])
        data['Gender'] = data['sample'].apply (lambda x: x.split (',')[1])
        data['Year'] = data['sample'].apply (lambda x: x.split (',')[2])
        data = data.drop (u'sample', 1)

        # reorder some columns
        cols = data.columns.tolist ()
        cols = cols[:4] + cols[-3:] + cols[4:-3]
        data = data[cols]
    elif 'diabetes' in filename:
        data = pd.melt (data, id_vars=data.columns.tolist ()[:2],
                        value_vars=data.columns.tolist ()[2:],
                        var_name=u'sample', value_name=u'percent')
        data['Sample Type'] = data['sample'].apply (lambda x: x.split (',')[0])
        data['Year'] = data['sample'].apply (lambda x: x.split (',')[1])
        data['Gender'] = data['sample'].apply (lambda x: x.split (',')[2])
        data = data.drop (u'sample', 1)

        # reorder some columns
        cols = data.columns.tolist ()
        cols = cols[:2] + cols[-3:] + cols[2:-3]
        data = data[cols]
    elif 'alcohol' in filename:
        data = data[data.columns.tolist ()[:-6]]
        data = data.rename (columns={'Location': 'County'})

        widgets = [progressbar.Percentage (), progressbar.Bar (),
                   progressbar.FormatLabel (
                       'Processed: %(value)d Counties (in: %(elapsed)s)')]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len (data))
        fips_getter = addfips.AddFIPS ()
        fips_vals = []
        for i in pbar (xrange (len (data))):
            try:
                fips = fips_getter.get_county_fips (data.iloc[i, :].loc['County'],
                                                    data.iloc[i, :].loc['State'])
            except:
                fips = np.nan
            fips_vals.append (fips)

        data['FIPS'] = fips_vals

        # reorder some columns
        cols = data.columns.tolist ()
        cols = cols[:2] + cols[-1:] + cols[2:-1]
        data = data[cols]

        data = pd.melt (data, id_vars=data.columns.tolist ()[:3],
                        value_vars=data.columns.tolist ()[3:],
                        var_name=u'sample', value_name=u'percent')
        data['Year'] = data['sample'].apply (lambda x: x.split (' ', 1)[0])
        data['Gender'] = data['sample'].apply (lambda x: x.split (' ', 1)[1])
        data = data.drop (u'sample', 1)
    elif 'SMOKING' in filename:
        widgets = [progressbar.Percentage (), progressbar.Bar (),
                   progressbar.FormatLabel (
                       'Processed: %(value)d Counties (in: %(elapsed)s)')]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len (data))
        fips_getter = addfips.AddFIPS ()
        fips_vals = []
        for i in pbar (xrange (len (data))):
            try:
                fips = fips_getter.get_county_fips (data.iloc[i, :].loc['county'],
                                                    data.iloc[i, :].loc['state'])
            except:
                fips = np.nan
            fips_vals.append (fips)

        data['FIPS'] = fips_vals

        # reorder some columns
        cols = data.columns.tolist ()
        cols = cols[:2] + cols[-1:] + cols[2:-1]
        data = data[cols]
    elif 'cdc' in filename:
        data = data.drop ('theme_range', 1)
        data['County'] = data['display_name'].apply (
            lambda x: x.strip ('"').split (',')[0] + ' County')
        data['State'] = data['display_name'].apply (
            lambda x: re.sub ('[() ]', '', x.strip ('"').split (',')[1]))

        def get_state_name (val):
            try:
                return (us.states.lookup (unicode (val)).name)
            except:
                return ('NA')

        data['State'] = data['State'].apply (get_state_name)
        data = data.drop (u'display_name', 1)

        cols = data.columns.tolist ()
        cols = cols[:2] + cols[-2:] + cols[2:-2]
        data = data[cols]
    elif 'food' in filename:
        def get_state_name (val):
            try:
                return (us.states.lookup (unicode (val)).name)
            except:
                return ('NA')

        def get_county_name (val):
            try:
                return (val + ' County')
            except:
                return (np.nan)

        data['County'] = data['County'].apply (get_county_name)
        data['State'] = data['State'].apply (get_state_name)

    data.to_sql (tablename, engine, if_exists=if_exists)

if __name__ == '__main__':

    data_dir = '/home/rmaunu/sandbox/insight/stroke_prev/data'

    dbname = 'stroke_prev'

    username = raw_input ('postgresql username: ')
    password = getpass.getpass ('password: ')

    engine = create_engine ('postgres://{0}:{1}@localhost/{2}'.format (
        username, password, dbname),
        pool_recycle=7200, pool_size=10)

    filename_bases = [
        # 'cdc_stroke_deaths_all_smoothed_plus_indicators',
        # 'cdc_stroke_deaths_all_plus_indicators',
        # 'cdc_stroke_hosp_65plus_smoothed',
        # 'cdc_stroke_hosp_65plus',
        # 'cdc_acute_myocard_infarc_deaths_all_smoothed',
        # 'cdc_all_heart_dis_deaths_all_smoothed',
        # 'cdc_card_dysrhyth_deaths_all_smoothed',
        # 'cdc_cor_heart_dis_deaths_all_smoothed',
        # 'cdc_hypertension_deaths_all_smoothed',
        # 'cdc_hypertension_deaths_all',
        # 'cdc_hypertension_hosp_65plus_smoothed',
        # 'cdc_hypertension_hosp_65plus',
        # 'IHME_USA_HYPERTENSION_BY_COUNTY_2001_2009',
        'IHME_US_COUNTY_TOTAL_AND_DAILY_SMOKING_PREVALENCE_1996_2012',
        # 'IHME_USA_OBESITY_PHYSICAL_ACTIVITY_2001_2011',
        # 'ihme_diabetes_diagnosed_1999_2012_Y2016M08D23',
        # 'ihme_diabetes_total_1999_2012_Y2016M08D23',
        # 'ihme_alcohol_use_any_2002_2012_Y2015M04D23',
        # 'ihme_alcohol_use_heavy_2002_2012_Y2015M04D23',
        # 'ihme_alcohol_use_binge_2002_2012_Y2015M04D23',
        # 'usda_food_access_feb2014',
        # 'usda_food_assistance_feb2014',
        # 'usda_food_health_feb2014',
        # 'usda_food_insecurity_feb2014',
        # 'usda_food_restaurants_feb2014',
        # 'usda_food_stores_feb2014',
        # 'pcen_v2015_y1015_txt',
    ]

    for base in filename_bases:
        print ('Making database: {0}, table: {1}'.format (dbname, base.lower ()))
        if 'pcen' in base:
            make_table ('{0}/{1}.txt'.format (data_dir, base),
                base.lower (), engine,
                if_exists='replace',
            )
        else:
            make_table ('{0}/{1}.csv'.format (data_dir, base),
                base.lower (), engine,
                if_exists='replace',
            )
