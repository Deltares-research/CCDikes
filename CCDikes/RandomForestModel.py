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
import rasterio
import numpy as np
import shap


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


def sample_dataframe_from_tif(dataframe, tiff_file):
    # open the tif file
    r = rasterio.open(tiff_file)
    # create deep copy of the dataframe
    values = dataframe.copy(deep=True)
    # change the crs of the dataframe
    values = values.to_crs("EPSG:28992")
    # get the points
    pts = np.array([values["geometry"].x, values["geometry"].y]).T
    # get the values of the points
    values_BAND = list(r.sample(pts))
    # flatten the list
    values_BAND = [item for sublist in values_BAND for item in sublist]
    return values_BAND

def sample_points_from_tif(pts, tiff_file):
    # open the tif file
    r = rasterio.open(tiff_file)
    # get the values of the points
    values_BAND = list(r.sample(pts))
    # flatten the list
    values_BAND = [item for sublist in values_BAND for item in sublist]
    return values_BAND


def preprocess_and_aggregate_data(database_path, layer_gpkg, add_satellite_data=False, time_section=30):
    punten = []
    for layer in layer_gpkg:
        if "punt" in layer:
            punten.append(gpd.read_file(database_path, layer=layer))
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
    fig2.show()
    # %%
    open_existing_file = True
    if open_existing_file:
            # load data geopandas geopackage
            gdf = gpd.read_file("../Data/inspection_points.gpkg", driver="GPKG", layer="inspection_points")
            #filtered_gdf = gdf[gdf["OBJECT_NAAM"] == "gras"]
            filtered_gdf = gdf

    else:
            filtered_gdf = check_meterological_data_per_point(filtered_gdf, time_section)
    # get also the year of the inspection
    # year is not needed for the model as it has no influence on the model
    #filtered_gdf["year"] = [date.year for date in filtered_gdf["DATUM"].to_list()]
    filtered_gdf['aspect'] = sample_dataframe_from_tif(filtered_gdf, "../data/aspect_area.tif")
    filtered_gdf['slope'] = sample_dataframe_from_tif(filtered_gdf, "../data/slope_area.tif")
    # -9999 is no data value
    # how many points have no data value
    print("Check if there are points with no data value for the slope and aspect variables")
    print(f"Number of points with no data value: {len(filtered_gdf[filtered_gdf['aspect'] == -9999])}")
    print(f"Number of points with no data value: {len(filtered_gdf[filtered_gdf['slope'] == -9999])}")
    # remove the points with no data value
    filtered_gdf = filtered_gdf[filtered_gdf['aspect'] != -9999]
    filtered_gdf = filtered_gdf[filtered_gdf['slope'] != -9999]

    # add the satellite data
    if add_satellite_data:
        # read the satellite data from the csv file
        satellite_data = pd.read_csv("../data/sentinel_2_data.csv")
        # keep only the nvdi values
        satellite_data = satellite_data[["ndvi", "x", "y"]]
        # to geodataframe
        satellite_data = gpd.GeoDataFrame(satellite_data, geometry=gpd.points_from_xy(satellite_data.x, satellite_data.y))
        # set the crs of the satellite data to the crs of the filtered_gdf
        satellite_data = satellite_data.set_crs("EPSG:4326")
        # intersect the inspection points with the satellite data points and merge the data
        filtered_gdf = gpd.sjoin_nearest(filtered_gdf, satellite_data, how="left")
        # check if there are nan values in the ndvi column
        print(f"Number of nan values in the ndvi column: {len(filtered_gdf[filtered_gdf['ndvi'].isna()])}")
        # remove the nan values
        filtered_gdf = filtered_gdf[filtered_gdf['ndvi'].notna()]
        drop_columns = ["index_right", "x_right", "y_right"]
        filtered_gdf = filtered_gdf.drop(drop_columns, axis=1)
        # rename the columns
        filtered_gdf = filtered_gdf.rename(columns={"x_left": "x", "y_left": "y"})
    full_dataset = filtered_gdf.copy()
    ## encode the categorical variables
    categorical_variables = ["OBJECT_NAAM",]
    drop_variables = [ "DATUM", "geometry", "KENMERK_NAAM", "PARAMETER_NAAM", "WAARDE"]
    # get the unique values of the categorical variables
    for variable in categorical_variables:
        print(f"{variable}: {filtered_gdf[variable].unique()}")
        # encode the categorical variables
        encoded = filtered_gdf[variable].astype('category').cat.codes
        # print the decoded and encoded values
        for i, value in enumerate(filtered_gdf[variable].unique()):
            print(f"{value}: {i}")
        # add the encoded values to the dataframe
        filtered_gdf[variable] = encoded
    # drop the variables
    filtered_gdf = filtered_gdf.drop(drop_variables, axis=1)
    output_labels = sorted(full_dataset['WAARDE'].unique().tolist())
    return filtered_gdf, output_labels

def perform_and_evaluate_random_forest(filtered_gdf, output_labels):
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
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)# Train the model on training data
    rf.fit(X_train, y_train)
    # calculate prediction accuracy for the initial model
    predictions = rf.predict(X_test) # Calculate the absolute errors
    # plot 1:1 line plot
    plt.hist(y_test.to_numpy())
    plt.hist(predictions)
    #plt.show()
    # %%
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=output_labels, cmap=plt.cm.Blues, normalize='true')
    #plt.show()
    accuracy = rf.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100.0}%")
    # %%
    feature_list = list(filtered_gdf.drop(outputs, axis=1).columns)
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    # plot shap values

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    # clean plots
    plt.show()
    plt.clf()
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.show()
    return rf, X_train, X_test, y_train, y_test

def plot_bbox_last_month(file_name):
    # read x y csv file
    x_y = pd.read_csv(file_name)
    xx = x_y['x'].to_numpy()
    yy = x_y['y'].to_numpy()
    # collect meteo data of the last 30 days
    time_section = 60
    #date = datetime.datetime.now()
    # 1rst of september 2023
    date = datetime.datetime(2022, 9, 1)
    # get the max and min dates
    end = date
    start = end - datetime.timedelta(days=time_section)
    # get the data
    results_meteo = KNMI(start=start, end=end, station="350", interval='day', variables=['TG', 'SQ', 'DR', 'RH', 'EV24'])
    mean_meteo = results_meteo.mean()
    # note that the data is the same for each point
    OBJECT_NAAM = 0 # 0 is the code for grass
    # get aspect and slope data
    points = np.array([xx.flatten(), yy.flatten()]).T
    aspect = sample_points_from_tif(points, "../data/aspect_area.tif")
    slope = sample_points_from_tif(points, "../data/slope_area.tif")
    # 28992 to 4326
    import pyproj
    transformer = pyproj.Transformer.from_crs("epsg:28992", "epsg:4326")
    # transform the points
    points = transformer.transform(points[:, 0], points[:, 1])

    # create a dataframe as the one used for the model
    df = pd.DataFrame({'OBJECT_NAAM': OBJECT_NAAM * np.ones(len(aspect)),
                       'x': points[0],
                       'y': points[1],
                       'TG': mean_meteo['TG'] * np.ones(len(aspect)),
                       'SQ': mean_meteo['SQ'] * np.ones(len(aspect)),
                       'DR': mean_meteo['DR'] * np.ones(len(aspect)),
                       'RH': mean_meteo['RH'] * np.ones(len(aspect)),
                       'EV24': mean_meteo['EV24'] * np.ones(len(aspect)),
                       'aspect': aspect,
                       'slope': slope,
                       })
    # to geo dataframe
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    # set the crs
    df.crs = "EPSG:4326"
    # read the ndvi data
    ndvi = gpd.read_file("../data/test_data_ndvi.csv")
    # set the geometry to the points x and y
    ndvi = ndvi.set_geometry(gpd.points_from_xy(ndvi.y, ndvi.x))
    # set the crs
    ndvi.crs = "EPSG:4326"
    # join the dataframes
    df = gpd.sjoin_nearest(df, ndvi, how="left")
    # columns to drop
    drop_columns = ndvi.columns.tolist()
    drop_columns.remove("ndvi")
    drop_columns.remove("x")
    drop_columns.remove("y")
    drop_columns.append("x_right")
    drop_columns.append("y_right")
    drop_columns.append("index_right")
    # drop the columns
    df = df.drop(drop_columns, axis=1)
    # rename x_left and y_left to x and y
    df = df.rename(columns={"x_left": "x", "y_left": "y"})
    # remove the points with no ndvi data that are nan
    # remove the points with no aspect or slope data
    df = df[df['aspect'] != -9999]
    df = df[df['slope'] != -9999]
    # get the predictions
    predictions = rf.predict(df)
    predictions = np.array(predictions)
    # create a dataframe
    df = pd.DataFrame({'x': df['y'], 'y': df['x'], 'predictions': predictions})
    # create predictions text according to the legend
    df['predictions'] = df['predictions'].replace(1, "Goed")
    df['predictions'] = df['predictions'].replace(2, "Redelijk")
    df['predictions'] = df['predictions'].replace(3, "Matig")
    df['predictions'] = df['predictions'].replace(4, "Slecht")

    # plot the results in a mapbox plot
    fig = px.scatter_mapbox(df, lat="y", lon="x", color="predictions", zoom=10, color_continuous_scale=px.colors.sequential.Rainbow)
    fig.update_layout(mapbox_style="open-street-map")
    fig.write_html('first_figure_summer.html', auto_open=True)



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
    # preprocess the data
    add_satellite_data = True
    filtered_gdf, output_labels = preprocess_and_aggregate_data(path_to_database,
                                                                layers,
                                                                add_satellite_data=add_satellite_data,
                                                                time_section=30)
    # perform and evaluate the random forest
    rf, X_train, X_test, y_train, y_test = perform_and_evaluate_random_forest(filtered_gdf, output_labels)
    # plot results in a contour plot of the area
    plot_bbox_last_month("test_cross_sections.csv")











