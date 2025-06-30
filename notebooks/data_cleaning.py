import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/Akshay/Projects/Hackathon Projects/Exoplanet_Habitability_Index_Predictor/datasets/exoplanet_dataset.csv")
df.head()

#drop unnecessary columns
drop_cols = ['P_DISCOVERY_FACILITY', 'P_YEAR', 'P_UPDATE', 'P_OMEGA', 'S_NAME_HD', 'S_NAME_HIP', 'S_RA', 
             'S_DEC', 'S_RA_STR', 'S_DEC_STR', 'P_INCLINATION', 'S_DISTANCE', 'P_DISTANCE', 'S_RA_TEXT', 
             'S_DEC_TEXT', 'S_CONSTELLATION_ABR', 'S_CONSTELLATION_ENG']
df.drop(columns = drop_cols, inplace = True, errors = 'ignore')

#Replace null values with the median value
median_impute_cols = ['P_MASS', 'P_RADIUS', 'P_PERIOD', 'P_SEMI_MAJOR_AXIS', 'P_ECCENTRICITY', 'S_MAG', 
                      'S_TEMPERATURE', 'S_MASS', 'S_RADIUS', 'S_METALLICITY', 'S_LOG_LUM', 'S_LOG_G',
                      'P_ESCAPE', 'P_POTENTIAL', 'P_GRAVITY', 'P_DENSITY', 'P_HILL_SPHERE', 'P_DISTANCE', 
                      'P_PERIASTRON', 'P_APASTRON', 'P_DISTANCE_EFF', 'P_FLUX', 'P_TEMP_EQUIL', 
                      'S_LUMINOSITY', 'S_SNOW_LINE', 'S_ABIO_ZONE', 'S_TIDAL_LOCK', 'S_AGE', 'P_TEMP_SURF']

for cols in median_impute_cols:
    if cols in df.columns:
        df[cols].fillna(df[cols].median(), inplace = True)

#Replace categorical values with the mode value
df['P_TYPE'].fillna(df['P_TYPE'].mode()[0], inplace=True)
df['S_TYPE'].fillna(df['S_TYPE'].mode()[0], inplace=True)
df['P_TYPE_TEMP'].fillna(df['P_TYPE_TEMP'].mode()[0], inplace=True)
df['S_TYPE_TEMP'].fillna(df['S_TYPE_TEMP'].mode()[0], inplace=True)

#Encoding categorical data in the dataset
categorical_data =  ['P_DETECTION', 'P_MASS_ORIGIN', 'S_TYPE', 'P_TYPE', 'S_TYPE_TEMP']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in categorical_data:
    df[col] = le.fit_transform(df[col])

from sklearn.preprocessing import OrdinalEncoder

ord_data = [['Cold', 'Warm', 'Hot']]
oe = OrdinalEncoder(categories= ord_data)
oe.fit(df[['P_TYPE_TEMP']])
df['P_TYPE_TEMP'] = oe.transform(df[['P_TYPE_TEMP']])

#Removing outliers using the IQR method
outliers = ['P_MASS', 'P_RADIUS', 'P_PERIOD', 'P_SEMI_MAJOR_AXIS', 'P_ECCENTRICITY', 'S_MAG', 'S_TEMPERATURE',
            'S_MASS', 'S_RADIUS', 'S_METALLICITY', 'S_LOG_LUM', 'S_LOG_G', 'P_ESCAPE', 'P_POTENTIAL', 
            'P_GRAVITY', 'P_HILL_SPHERE', 'P_PERIASTRON', 'P_APASTRON', 'P_FLUX', 'P_TEMP_SURF', 'S_LUMINOSITY',
            'S_ABIO_ZONE', 'S_TIDAL_LOCK']

mask = pd.Series(True, index=df.index)

for col in outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask &= df[col].between(lower, upper)

df = df[mask]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

col_names = ['P_MASS', 'P_RADIUS', 'P_PERIOD', 'P_SEMI_MAJOR_AXIS', 'P_ECCENTRICITY', 'S_MAG', 'S_TEMPERATURE', 
             'S_MASS', 'S_RADIUS', 'S_METALLICITY', 'S_AGE', 'S_LOG_LUM', 'S_LOG_G', 'P_ESCAPE', 'P_POTENTIAL', 
             'P_GRAVITY', 'P_DENSITY', 'P_HILL_SPHERE', 'P_PERIASTRON', 'P_APASTRON', 'P_DISTANCE_EFF', 
             'P_FLUX', 'P_TEMP_EQUIL', 'P_TEMP_SURF', 'S_TYPE_TEMP', 'S_LUMINOSITY', 'S_SNOW_LINE', 
             'S_ABIO_ZONE', 'S_TIDAL_LOCK', 'P_HABZONE_OPT', 'P_HABZONE_CON']

for col in col_names:
    df[col] = scaler.fit_transform(df[[col]]).ravel()

df.drop_duplicates(inplace=True)

df.to_csv('C:/Users/aksha/OneDrive/Desktop/Akshay/Projects/Hackathon Projects/Exoplanet_Habitability_Index_Predictor/datasets/exoplanet_dataset_cleaned.csv', index=False)
