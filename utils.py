#!/usr/bin/env python

import cPickle as pickle
import os
import socket
import time

features_key = {
    'cnty_fips': 'FIPS',
    'County': 'County',
    'State': 'State',
    'dm_prev_adj': 'Diagnosed Diabetes (%)',
    'ob_prev_adj': 'Obesity Percentage (%)',
    'ltpia_prev_adj': 'Physical Inactivity Percentage (%)',
    'no_hsdip': 'No HS Diploma (%)',
    'no_college': 'No College Diploma (%)',
    'female_hd': 'Female-Headed Household (%)',
    'foodstmp': 'SNAP Recipients (%)',
    'home_val': 'Median Home Value ($1000s)',
    'income': 'Median Household Income ($1000s)',
    'GINI': 'GINI Coefficient (Income Inequality)',
    'povpct': 'Population Below Poverty Line (%)',
    'unemploy': 'Unemployment Rate (%)',
    'perc_aian': 'American Indian/Alaska Native (%)',
    'perc_api': 'Asian/Pacific Islander (%)',
    'perc_black': 'African American (%)',
    'perc_white': 'Non-Hispanic White (%)',
    'perc_hisp': 'Hispanic (%)',
    'perc_65up': '65yr or older (%)',
    'total_pop': 'Total Population',
    'log10_total_pop': 'Total Population',
    'airqual': 'Air Quality PM2.5',
    'PARKS': 'Percentage of Population Living Within Half a Mile of a Park',
    'sev_housing': 'Percentage of Households Living with Severe Housing Problems, 2008-2012',
    'NCHS_collapse': 'Urban-Rural Status',
    'htnadh_all': 'Blood-Pressure Medication Nonadherence (%)',
    'htnadh_white': 'Blood-Pressure Medication Nonadherence, Non-Hispanic White (%)',
    'htnadh_black': 'Blood-Pressure Medication Nonadherence, African American (%)',
    'htnadh_hisp': 'Blood-Pressure Medication Nonadherence, Hispanic (%)',
    'htnadh_api': 'Blood-Pressure Medication Nonadherence, Asian/Pacific Islander (%)',
    'htnadh_aian': 'Blood-Pressure Medication Nonadherence, American Indian/Alaska Native (%)',
    'diuradh': 'Diuretic Nonadherence (%)',
    'rasadh': 'Renin-Angiotensin System Antagonist Nonadherence (%)',
    'stg_hosp': 'Hospitals',
    'stg_int_cr': 'Hospitals with Intensive-Care Unit',
    'stg_rehab': 'Hospitals with Cardiac-Rehabilitation Unit',
    'stg_emerg': 'Hospitals with Emergency Department',
    'stg_neuro': 'Hospitals with Neurological Sevices',
    'pharmpc': 'Pharmacies and Drug-Stores (per 100,000)',
    'prim_dr': 'Population per Primary-Care Physician',
    'cvd_dr': 'Population per Cardiovascular Disease Physician',
    'neuro_dr': 'Population per Neurologist',
    'surg_dr': 'Population per Neurosurgeon',
    'pctui': 'Percentage without Health Insurance, Under Age 65',
    'medicaid': 'Medicade Eligible (%)',
    'stroke_hosp': 'Stroke Hosp. Rate (per 1000 Medicare Ben.)',
    'all_heart_dis_hosp': 'All Heart Disease Hospitalization Rate (per 1000 Medicare Ben.)',
    'cor_heart_dis_hosp': 'Coronary Heart Disease Hospitalization Rate (per 1000 Medicare Ben.)',
    'hyperten_hosp': 'Hypertension Hospitalization Rate (per 1000 Medicare Ben.)',
    'acute_myocard_infarc_hosp': 'Acute Myocardial Infarction Hospitalization Rate (per 1000 Medicare Ben.)',
    'card_dysrhytm_hosp': 'Cardiac Dysrhythmia Hospitalization Rate (per 1000 Medicare Ben.)',
    'stroke_death': 'Stroke Mortality Rate (per 100000)',
    'all_heart_dis_death': 'Heart Disease Mortality Rate (per 100000)',
    'cor_heart_dis_death': 'Coronary Heart Disease Mortality Rate (per 100000)',
    'hyperten_death': 'Hypertension Mortality Rate (per 100000)',
    'acute_myocard_infarc_death': 'Acute Myocardial Infarction Mortality Rate (per 100000)',
    'any_alcohol_2011': 'Prevalence of Any Drinking (%)',
    'binge_alcohol_2011': 'Prevalence of Binge Drinking (%)',
    'heavy_alcohol_2011': 'Prevalence of Heavy Drinking (%)',
    'total_mean_smoking_2011': 'Total Smoking Prevalence (%)',
    'daily_mean_smoking_2011': 'Daily Smoking Prevalence (%)',
    'snap_redemp_per_store_2012': 'SNAP redemptions per SNAP-authorized stores (2012)',
    'pct_obese_adults_2013': 'Obese Adult Prevalence (%, 2013)',
    'rec_fac_2012': 'Recreation & fitness facilities (per 1,000, 2012)',
    'ers_nat_amenity_index_1999': 'ERS natural amenity index (1999)',
    'food_insec_house_pct_10_12': 'Household Food Insecurity (%, 2010-2012)',
    'low_access_food_pct10': 'Population, low access to store (%, 2010)',
    'low_access_grocery_pct10': 'Grocery Stores (per 1000, 2010)',
    'grocery_pct10': 'Grocery Stores (per 1000, 2010)',
    'low_access_food_low_inc_pct10_norm': 'Fraction low access to store, low income (2010)',
    'low_access_food_snr_pct10_norm': 'Fraction low access to store, seniors (2010)',
    'daily_mean_smoking_2011_norm': 'Fraction of smokers, daily (2011)',
    'binge_alcohol_2011_norm': 'Fraction of drinking, binge (2011)',
    'heavy_alcohol_2011_norm': 'Fraction of drinking, heavy (2011)',
    'diuradh_norm': 'Fraction of BLMN, Diuretic',
    'rasadh_norm': 'Fraction of BLMN, Renin-Angiotensin System Antagonist',
}

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

def strip_county_formatting (county):
    return (county.replace ('-', ' ').replace ("'", "").replace (
        ".", "").replace (",", "").lower ().replace ("saint", "st").replace (
        " county", ""))


def ensure_dir (dirname):
    """Make sure ``dirname`` exists and is a directory."""

    if not os.path.isdir (dirname):
        try:
            os.makedirs (dirname)   # throws if exists as file
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
    return dirname


def save (obj, filename):
    """Dump `obj` to `filename` using the pickle module."""

    outdir, outfile = os.path.split (filename)
    save_id = '{0}_nixtime_{2:.0f}_job_{1}'.format (
        socket.gethostname (), os.getpid (), time.time ())
    temp_filename = os.path.join (outdir, '.part_{0}_id_{1}'.format (
        outfile, save_id))
    with open (temp_filename, 'wb') as f:
        pickle.dump (obj, f, -1)
    os.rename (temp_filename, filename)


def resave (obj):
    """Dump `obj` to the filename from which it was loaded."""

    save (obj, obj.__cache_source_filename)


def load (filename):
    """Load `filename` using the pickle module."""

    with open (filename) as f:
        out = pickle.load (f)
        try:
            out.__cache_source_filename = filename
        except:
            pass
        return out
