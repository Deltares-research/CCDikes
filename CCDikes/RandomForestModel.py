from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import datetime
import requests
import json
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def KNMI(start, end, station="348", interval='hour', variables=['RH', 'FF', 'RH']):
    r"""
    Function to download KNMI data from the KNMI website.

    start: start date in the format YYYYMMDD
    end: end date in the format YYYYMMDD
    station: station number (default: 348)
    interval: interval of the data (default: hour)
    DD: Windrichting (in graden) gemiddeld over de laatste 10 minuten van het
        afgelopen uur (360=noord, 90=oost, 180=zuid, 270=west, 0=windstil 990=veranderlijk).
    FF: Uurgemiddelde windsnelheid (in 0.1 m/s)
    RH: Uursom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm)
    returns: pandas dataframe with the data
    """

    Schemas = {"start": start,
               "end": end,
               "fmt": "json",
               "stns": station}

    if interval == 'hour':
        response = requests.post(r"https://www.daggegevens.knmi.nl/klimatologie/uurgegevens", data=Schemas)
    elif interval == 'day':
        response = requests.post(r"https://www.daggegevens.knmi.nl/klimatologie/daggegevens", data=Schemas)

    data = json.loads(response.content)

    df = pd.DataFrame(data)
    if df.empty:
        print('No data available for this period.')
        return None

    if interval == 'hour':
        df['datetime'] = df['date'].astype('str').str[:10] + 'T' + (df['hour'] - 1).astype('str') + ':00:00'
        df['datetime'] = df['datetime'].astype('datetime64[ns]') + pd.DateOffset(hours=1)
    elif interval == 'day':
        df['datetime'] = pd.to_datetime(df['date'])

    df = df[['datetime'] + variables]

    df.RH = df.RH.replace(-1, 0)  # Get rid of -1 values
    df.RH = df.RH / 10000  # Get to SI unit m

    try:
        df.FF = df.FF / 10  # Get to SI unit m/s
    except:
        pass
    try:
        df.FF = df.FHX / 10  # Get to SI unit m/s
    except:
        pass
    try:
        df.EV24 = df.EV24 / 10000  # Get to SI unit m
    except:
        pass

    return df


def check_meterological_data_per_point(punten, time_section):
    # check the meterological data
    # get the unique dates
    dates = punten["DATUM"].to_list()
    meteo_results = []
    for date in dates:
        # get the max and min dates
        end = date
        start = end - datetime.timedelta(days=time_section)
        # get the data
        df = KNMI(start=start, end=end, station="350", interval='day', variables=['TG', 'SQ', 'DR', 'RH', 'EV24'])
        if df is None:
            meteo_results.append(pd.DataFrame({"TG": 0, "SQ": 0, "DR": 0, "RH": 0, "EV24": 0}))
        # get average values of the dataframe per column
        meteo_results.append(df[['TG', 'SQ', 'DR', 'RH', 'EV24']].sum(axis=0))
    # add the results to the dataframe
    punten["TG"] = [result["TG"] for result in meteo_results]
    punten["SQ"] = [result["SQ"] for result in meteo_results]
    punten["DR"] = [result["DR"] for result in meteo_results]
    punten["RH"] = [result["RH"] for result in meteo_results]
    punten["EV24"] = [result["EV24"] for result in meteo_results]
    return punten



if __name__ == "__main__":
    # read the geo database
    path_to_database = "..\data\Digispectie2017_2022\Digispectie2017_2022.gdb"
    layers = ['Inspectie2017_najaar_punt',
              'Inspectie2018_najaar_punt',
              'Inspectie2019_najaar_punt',
              'Inspectie2020_voorjaar_punt',
              'Inspectie2020_najaar_punt',
              'Inspectie2021_voorjaar_punt',
              'Inspectie2021_najaar_punt',
              'Inspectie2022_voorjaar_punt']
    # %%
    punten = []
    for layer in layers:
        if "punt" in layer:
            punten.append(gpd.read_file(path_to_database, layer=layer))
    punten = gpd.GeoDataFrame(pd.concat(punten, axis=0, sort=False))
    # %%
    punten["inspection_value"] = [int(value.split("_")[0]) for value in punten["WAARDE"].to_list()]
    # %%
    fig = px.parallel_categories(punten,
                                     dimensions=["OBJECT_NAAM", "PARAMETER_NAAM",  "WAARDE"],
                                     color="inspection_value",
                                     color_continuous_scale=px.colors.sequential.YlGnBu_r)
    fig.show()
    # keep only "grass" of OBJECT_NAAM
    filtered_gdf = punten[punten["OBJECT_NAAM"] == "gras"]
    # %%
    fig2 = px.parallel_categories(filtered_gdf,
                                     dimensions=["OBJECT_NAAM", "PARAMETER_NAAM", "WAARDE"],
                                     color="inspection_value",
                                     color_continuous_scale=px.colors.sequential.YlGnBu_r)
    fig.show()
    # %%
    open_existing_file = True
    if open_existing_file:
            # load data geopandas geopackage
            gdf = gpd.read_file("../Data/inspection_points.gpkg", driver="GPKG", layer="inspection_points")
            #filtered_gdf = gdf[gdf["OBJECT_NAAM"] == "gras"]
            filtered_gdf = gdf

    else:
            filtered_gdf = check_meterological_data_per_point(filtered_gdf, 30)

    full_dataset = filtered_gdf.copy()
    ## encode the categorical variables
    categorical_variables = ["OBJECT_NAAM",]
    drop_variables = [ "DATUM", "geometry", "KENMERK_NAAM", "PARAMETER_NAAM", "WAARDE"]
    # get the unique values of the categorical variables
    for variable in categorical_variables:
        print(f"{variable}: {filtered_gdf[variable].unique()}")
        # encode the categorical variables
        filtered_gdf[variable] = filtered_gdf[variable].astype('category').cat.codes
    # drop the variables
    filtered_gdf = filtered_gdf.drop(drop_variables, axis=1)
    output_labels = sorted(full_dataset['WAARDE'].unique().tolist())

    # %%
    ## split the data into train and test data
    outputs = ["inspection_value"]
    X_train, X_test, y_train, y_test = train_test_split(filtered_gdf.drop(outputs, axis=1), filtered_gdf[outputs], test_size=0.2, random_state=42)
    # %%
    # print the shapes of the data
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    # %%
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)# Train the model on training data
    rf.fit(X_train, y_train)
    # calculate prediction accuracy for the initial model
    predictions = rf.predict(X_test) # Calculate the absolute errors
    # plot 1:1 line plot
    plt.hist(y_test.to_numpy())
    plt.hist(predictions)
    plt.show()
    # %%
    ConfusionMatrixDisplay.from_predictions( y_test, predictions, display_labels=output_labels, cmap=plt.cm.Blues, normalize='true')
    accuracy = rf.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100.0}%")
    # %%
    feature_list = list(filtered_gdf.drop(outputs, axis=1).columns)
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=30)
    df = KNMI(start=start, end=end, station="350", interval='day', variables=['TG', 'SQ', 'DR', 'RH', 'EV24'])
    df.dropna(inplace=True)
    df = df[['TG', 'SQ', 'DR', 'RH', 'EV24']]
    df_sum = df.sum(axis=0)


