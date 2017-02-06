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
    'perc_65up': 'Population, 65yr or older (%)',
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
    'prim_dr': 'Population per Primary-Care Physician (1,000s)',
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
    'heavy_alcohol_2011': 'Prevalence of Heavy Drinking (%)',
    'total_mean_smoking_2011': 'Total Smoking Prevalence (%)',
    'daily_mean_smoking_2011': 'Daily Smoking Prevalence (%)',
    'snap_redemp_per_store_2012': 'SNAP redemptions per SNAP-authorized stores (2012)',
    'pct_obese_adults_2013': 'Obese Adult Prevalence (%, 2013)',
    'rec_fac_2012': 'Recreation & fitness facilities (per 1,000, 2012)',
    'ers_nat_amenity_index_1999': 'ERS natural amenity index (1999)',
    'food_insec_house_pct_10_12': 'Household Food Insecurity (%, 2010-2012)',
    'low_access_food_pct10': 'Population, low access to store (%, 2010)',
    'low_access_grocery_pct10': 'Population, low access to grocery store (%, 2010)'
}

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
