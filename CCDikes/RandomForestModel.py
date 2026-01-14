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

    if "RH" in variables:

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
        df = KNMI(start=start, end=end, station="350", interval='day', variables=[ 'EV24'])
        if df is None:
            meteo_results.append(pd.DataFrame({ "EV24": 0}))
        # get average values of the dataframe per column
        meteo_results.append(df[['EV24']].sum(axis=0))
    # add the results to the dataframe
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
    # statistics of the data
    data_list = ['OBJECT_NAAM', 'PARAMETER_NAAM', 'WAARDE', 'inspection_value']
    for data in data_list:
        # print the type unique values
        print(f"Unique values of {data}: {punten[data].unique()}")
        # plot the data
        fig = px.histogram(punten, x=data, color="WAARDE",)
        fig.write_html(f"fig_{data}.html")

    # %%
    fig = px.parallel_categories(punten,
                                     dimensions=["OBJECT_NAAM", "PARAMETER_NAAM",  "WAARDE"],
                                     color="inspection_value",
                                     color_continuous_scale=px.colors.sequential.YlGnBu_r)
    fig.write_html("fig_all_data.html")
    # keep only "grass" of OBJECT_NAAM
    filtered_gdf = punten[punten["OBJECT_NAAM"] == "gras"]
    # %%
    fig2 = px.parallel_categories(filtered_gdf,
                                     dimensions=["OBJECT_NAAM", "PARAMETER_NAAM", "WAARDE"],
                                     color="inspection_value",
                                     color_continuous_scale=px.colors.sequential.YlGnBu_r)
    fig2.write_html("fig_grass_data.html")
    # %%
    open_existing_file = True
    if open_existing_file:
            # load data geopandas geopackage
            gdf = gpd.read_file("../Data/inspection_points.gpkg", driver="GPKG", layer="inspection_points")
            #filtered_gdf = gdf[gdf["OBJECT_NAAM"] == "gras"]
            # remove columns
            remove_columns = ["TG", "SQ", "DR", "RH"]
            gdf = gdf.drop(columns=remove_columns)
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
    categorical_variables = ["OBJECT_NAAM","PARAMETER_NAAM"]
    drop_variables = [ "DATUM", "geometry", "KENMERK_NAAM", "WAARDE", "x", "y"]
    # group of remove values in the PARAMETER_NAAM column
    group_1 = ['onkruid klein','kale plekken','natte plekken','sterkte graszode','onkruid groot','ruigte of houtopslag','bedekkingsgraad','bedekkingsgraad oeverlijn',]
    group_2 = ['scheuren', 'zanduitspoeling','aansluiting','aansluiting grondlichaam (afkalving)','erosieafslag',]
    group_3 = ['graverij klein','verzakkingen of opbollingen','graverij groot','spoorvorming','verzakkingen']
    remove_values = group_1 + group_2 + group_3
    # remove the values from the PARAMETER_NAAM column that are not in the remove_values list
    filtered_gdf = filtered_gdf[filtered_gdf["PARAMETER_NAAM"].isin(remove_values)]
    # group the values in the PARAMETER_NAAM column
    filtered_gdf.loc[filtered_gdf["PARAMETER_NAAM"].isin(group_1), "PARAMETER_NAAM"] = "group_1"
    filtered_gdf.loc[filtered_gdf["PARAMETER_NAAM"].isin(group_2), "PARAMETER_NAAM"] = "group_2"
    filtered_gdf.loc[filtered_gdf["PARAMETER_NAAM"].isin(group_3), "PARAMETER_NAAM"] = "group_3"

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
    # save the plot in high resolution
    plt.savefig("confusion_matrix.svg", dpi=1200, format='svg')
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
    # rename output labels
    output_labels = ['Goed', 'Redelijk', 'Matig', 'Slecht']

    # clean plots
    plt.show()
    plt.clf()
    # plot but simple bar plot with shap values and square format
    shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=output_labels, show=False, plot_size=(10, 10))
    # change font size
    plt.rcParams.update({'font.size': 44})
    # save the plot in high resolution
    plt.savefig("shap_values.svg", dpi=1200, format='svg')
    return rf, X_train, X_test, y_train, y_test


def plot_bbox_last_month(file_name, rf):
    # read x y csv file
    x_y = pd.read_csv(file_name)
    xx = x_y['x'].to_numpy()
    yy = x_y['y'].to_numpy()
    # collect meteo data of the last 30 days
    time_section = 60
    #date = datetime.datetime.now()
    # 1rst of september 2023
    date = datetime.datetime(2022, 6, 1)
    # get the max and min dates
    end = date
    start = end - datetime.timedelta(days=time_section)
    # get the data
    results_meteo = KNMI(start=start, end=end, station="350", interval='day', variables=[ 'EV24'])
    mean_meteo = results_meteo.mean()
    # note that the data is the same for each point
    OBJECT_NAAM = 0 # 0 is the code for grass
    # get aspect and slope data
    points = np.array([xx.flatten(), yy.flatten()]).T
    # points to 28992
    import pyproj
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:28992")
    new_points = transformer.transform(points[:, 1], points[:, 0])
    new_points = np.array(new_points).T
    aspect = sample_points_from_tif(new_points, "../data/aspect_area.tif")
    slope = sample_points_from_tif(new_points, "../data/slope_area.tif")
    # 28992 to 4326

    # create a dataframe as the one used for the model
    df = pd.DataFrame({'OBJECT_NAAM': OBJECT_NAAM * np.ones(len(aspect)),
                       'x': points[:, 1],
                       'y': points[:, 0],
                       'EV24': mean_meteo['EV24'] * np.ones(len(aspect)),
                       'aspect': aspect,
                       'slope': slope,
                       })
    # to geo dataframe
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    # set the crs
    df.crs = "EPSG:4326"
    # read the ndvi data
    ndvi = gpd.read_file("../data/test_data_ndvi_summer.csv")
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
    # remove x and y
    ds_no_xy = df.drop(['x', 'y'], axis=1)
    predictions = rf.predict(ds_no_xy)
    predictions = np.array(predictions)
    # create a dataframe
    df = pd.DataFrame({'x': df['y'], 'y': df['x'], 'predictions': predictions})
    # create predictions text according to the legend
    df['predictions'] = df['predictions'].replace(1, "Goed")
    df['predictions'] = df['predictions'].replace(2, "Redelijk")
    df['predictions'] = df['predictions'].replace(3, "Matig")
    df['predictions'] = df['predictions'].replace(4, "Slecht")
    # define the colors for the legend
    colors = {'Goed': 'green', 'Redelijk': 'yellow', 'Matig': 'orange', 'Slecht': 'red'}

    # plot the results in a mapbox plot
    fig = px.scatter_mapbox(df, lat="y", lon="x", color="predictions", color_discrete_map=colors, zoom=12)
    fig.update_layout(mapbox_style="open-street-map")
    fig.write_html('first_figure_summer.html', auto_open=True)
    # to shapefile
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    df['predictions'] = df['predictions'].replace("Goed", 1)
    df['predictions'] = df['predictions'].replace("Redelijk", 2)
    df['predictions'] = df['predictions'].replace("Matig", 3)
    df['predictions'] = df['predictions'].replace("Slecht", 4)
    df.crs = "EPSG:4326"
    df.to_file("predictions_summer.shp")


def test_classification_model(model_class, directory):
    import os
    import pyproj
    # get all csv files in the directory that contain the word "dense"
    csv_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and "dense" in f]
    # make one big dataframe
    dfs = []
    for file in csv_files:
        dfs.append(pd.read_csv(directory + file))
    # get the features
    merged = pd.concat(dfs)


    time_section = 60
    #date = datetime.datetime.now()
    # 1rst of september 2023
    date = datetime.datetime(2022, 9, 1)
    # get the max and min dates
    end = date
    start = end - datetime.timedelta(days=time_section)
    # get the data
    results_meteo = KNMI(start=start, end=end, station="350", interval='day', variables=[ 'EV24'])
    mean_meteo = results_meteo.mean()
    # note that the data is the same for each point
    OBJECT_NAAM = 0 # 0 is the code for grass
    # get aspect and slope data
    points = np.array([merged['x'], merged['y']]).T
    # transform the points from 4326 to 28992
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:28992")
    # transform the points
    points = transformer.transform(points[:, 1], points[:, 0])
    points = np.array(points).T
    aspect = sample_points_from_tif(points, "../data/aspect_area.tif")
    slope = sample_points_from_tif(points, "../data/slope_area.tif")
    # 28992 to 4326

    transformer = pyproj.Transformer.from_crs("epsg:28992", "epsg:4326")
    # transform the points
    points = transformer.transform(points[:, 0], points[:, 1])

    # create a dataframe as the one used for the model
    df = pd.DataFrame({'OBJECT_NAAM': OBJECT_NAAM * np.ones(len(aspect)),
                       'x': points[0],
                       'y': points[1],
                       'EV24': mean_meteo['EV24'] * np.ones(len(aspect)),
                       'aspect': aspect,
                       'slope': slope,
                       })
    # remove the points with no aspect or slope data
    df = df[df['aspect'] != -9999]
    df = df[df['slope'] != -9999]
    # keep points
    keep_points = df[['x', 'y']]
    # to geo dataframe
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    # set the crs
    df.crs = "EPSG:4326"
    # read the ndvi data
    ndvi = merged
    # set the geometry to the points x and y
    ndvi = ndvi.set_geometry(gpd.points_from_xy(ndvi.y, ndvi.x))
    # set the crs
    ndvi.crs = "EPSG:4326"
    # join the dataframes
    df = gpd.sjoin_nearest(df, ndvi, how="left")
    # columns to keep
    keep_columns = [
        'EV24',
        'aspect',
        'slope',
        'ndvi',

    ]
    df = df[keep_columns]


    # pass the dataframe to the model
    prediction = model_class.predict(df)
    # get the predictions probabilities
    prediction_probabilities = model_class.predict_proba(df)
    # add x and y to the dataframe
    df['x'] = keep_points['x']
    df['y'] = keep_points['y']
    # decode the predictions
    group = {0: "Graverij", 1: "Gras", 2: "Scheren"}
    prediction = [group[prediction[i]] for i in range(len(prediction))]
    # add the predictions to the dataframe
    df['Groep'] = prediction
    # plot the results in a mapbox plot
    fig = px.scatter_mapbox(df, lat="x", lon="y", color='Groep', zoom=12)
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()
    # plot the probabilities in a mapbox plot with three traces
    fig = px.scatter_mapbox(df, lat="x", lon="y", color=prediction_probabilities[:, 0], zoom=12)
    # add title
    fig.update_layout(title_text="Graverij potentieel")
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()
    fig = px.scatter_mapbox(df, lat="x", lon="y", color=prediction_probabilities[:, 1], zoom=12)
    # add title
    fig.update_layout(title_text="Gras potentieel")
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()
    fig = px.scatter_mapbox(df, lat="x", lon="y", color=prediction_probabilities[:, 2], zoom=12)
    # add title
    fig.update_layout(title_text="Scheren potentieel")
    fig.update_layout(mapbox_style="open-street-map")
    fig.show()


def accuracy_score(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)

def plot_confusion_matrix(y_true, y_pred, features):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    # get the confusion matrix in percentages
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    # plot the confusion matrix
    plt.figure(figsize=(10, 7))
    # change the labels
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    # change the labels in ticks in the middle of the squares
    loc = np.arange(len(features)) + 0.5
    plt.xticks( loc, features)
    plt.yticks(loc, features)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def perform_and_evaluate_xboost(df, labels):
    import xgboost as xgb
    outputs = "inspection_value"
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(df.drop(outputs, axis=1), df[outputs], test_size=0.2, random_state=42)
    # create the xgboost model
    model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    # fit the model
    model.fit(X_train.to_numpy(), y_train.to_numpy() - 1)
    # get the predictions
    predictions = model.predict(X_test.to_numpy())
    # get the accuracy
    accuracy = accuracy_score(y_test - 1, predictions)
    print("Accuracy of the model is {}".format(accuracy))
    # get the feature importance
    feature_importance = model.feature_importances_
    # get the feature names
    feature_names = df.drop(outputs, axis=1).columns.tolist()
    # create a dataframe
    df_feature_importance = pd.DataFrame({'feature_names': feature_names, 'feature_importance': feature_importance})
    # sort the dataframe
    df_feature_importance = df_feature_importance.sort_values(by=['feature_importance'], ascending=False)
    # plot the feature importance
    fig = px.bar(df_feature_importance, x='feature_names', y='feature_importance')
    fig.show()
    # plot the confusion matrix
    plot_confusion_matrix( y_test, predictions, feature_names)
    plt.show()


def perform_information_gain_analysis(df, label):
    from sklearn.feature_selection import mutual_info_classif
    # get the information gain
    information_gain = mutual_info_classif(df.drop([label], axis=1).to_numpy(), df[label].to_numpy() - 1)
    # create a dataframe
    df_information_gain = pd.DataFrame({'feature_names': df.drop([label], axis=1).columns.tolist(), 'information_gain': information_gain})
    # sort the dataframe
    df_information_gain = df_information_gain.sort_values(by=['information_gain'], ascending=False)
    # plot the information gain
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(df_information_gain['feature_names'], df_information_gain['information_gain'])
    ax.set_xticklabels(df_information_gain['feature_names'], rotation=90)
    ax.set_title("Information gain")
    plt.savefig("information_gain.png")
    # also do simple correlation analysis for the label
    df_corr = df.corr()
    df_corr = df_corr.sort_values(by=[label], ascending=False)
    # plot the correlation do not plot the label itself
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(df_corr.index, df_corr[label])
    ax.set_xticklabels(df_corr.index, rotation=45)
    # set the y axis
    ax.set_ylim([-1.2, 1.2])
    # set labels
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Correlation")
    plt.savefig("correlation.png")

def classification_with_tensorflow(df, output):
    from ngboost import NGBClassifier
    from ngboost.distns import k_categorical, Bernoulli
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(df.drop([output, 'inspection_value'] + ['OBJECT_NAAM'], axis=1), df[output], test_size=0.2, random_state=42)
    # create the model
    n_of_output_classes = len(df[output].unique())
    ngb_cat = NGBClassifier(Dist=k_categorical(n_of_output_classes),
                            verbose=True,
                            random_state=42,
                            n_estimators=10000,
                            learning_rate=0.05,
                            minibatch_frac=0.5,
                            col_sample=0.5,)
    # fit the model
    ngb_cat.fit(X_train.to_numpy(), y_train.to_numpy())
    # plot the confusion matrix
    predictions_det = ngb_cat.predict(X_test.to_numpy())
    plot_confusion_matrix(y_test, predictions_det, ['graverij', 'gras', 'scheuren'])
    plt.show()
    predictions_prob = ngb_cat.predict_proba(X_test.to_numpy())
    # plot the probabilities
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(predictions_prob[:, 0], bins=50, alpha=0.5, label='graverij')
    ax.hist(predictions_prob[:, 1], bins=50, alpha=0.5, label='gras')
    ax.hist(predictions_prob[:, 2], bins=50, alpha=0.5, label='scheuren')
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability distribution")
    ax.legend()
    plt.show()
    return ngb_cat






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


    #classify_model = classification_with_tensorflow(filtered_gdf, "PARAMETER_NAAM")
    # drop the columns that are not needed
    filtered_gdf = filtered_gdf.drop(['PARAMETER_NAAM'], axis=1)
    #test_classification_model(classify_model, "D:/CCDikes/data/")
    # check correlations between the features
    #perform_information_gain_analysis(filtered_gdf, "OBJECT_NAAM")
    # perform and evaluate the random forest
    rf, X_train, X_test, y_train, y_test = perform_and_evaluate_random_forest(filtered_gdf, output_labels)
    # merge all csv files

    # find all csv files in the folder
    import os
    folder = "D:\CCDikes\data"
    #csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv") and "dense" in f]
    # merge all csv files
    #df = pd.concat([pd.read_csv(f) for f in csv_files])
    # write the merged csv file
    #df.to_csv(folder + "\merged.csv", index=False)
    # plot results in a contour plot of the area
    plot_bbox_last_month(folder + "/test_data_ndvi_summer.csv", rf)











