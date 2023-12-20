import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import requests


def preprocess_and_aggregate_data(database_path, layer_gpkg):
    punten = []
    for layer in layer_gpkg:
        if "punt" in layer:
            punten.append(gpd.read_file(database_path, layer=layer))
    punten = gpd.GeoDataFrame(pd.concat(punten, axis=0, sort=False))
    # %%
    punten["inspection_value"] = [int(value.split("_")[0]) for value in punten["WAARDE"].to_list()]
    # define x and y columns but in latlon
    punten = punten.set_crs("EPSG:28992")
    # change crs
    punten = punten.to_crs("EPSG:4326")
    # get x and y as columns
    punten["lon"] = punten.geometry.x
    punten["lat"] = punten.geometry.y

    return punten


def group_per_value(value):
    inspections = {}
    for name, group in punten_group:
        if len(group) in inspections.keys():
            inspections[len(group)] += group[value].to_list()
        else:
            inspections[len(group)] = group[value].to_list()
    # plot how many inspections per location
    length_subgroups = len(inspections.keys())
    fig = make_subplots(rows=length_subgroups, cols=1, subplot_titles=[f"Number of inspections per location: {key}" for key in inspections.keys()])
    for index, key in enumerate(inspections.keys()):
        fig.add_trace(
            go.Histogram(x=inspections[key], name=f"Number of inspections per location: {key}"),
            row=index+1, col=1
        )
    # add spacing between the subplots
    fig.update_layout(height=1000,showlegend=False)
    fig.update_layout(title_text="Number of inspections per location")
    fig.write_html("../results/number_of_inspections_per_location.html")
    fig.show()


def KNMI_jaar_overzicht(start_year, end_year):
    url_percipitaction = "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/maandgegevens/mndgeg_344_rh24.txt"
    url_temp_min = "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/maandgegevens/mndgeg_344_tng.txt"
    url_temp_max = "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/maandgegevens/mndgeg_344_txg.txt"
    url_temp_mean = "https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/maandgegevens/mndgeg_344_tg.txt"


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
    punten = preprocess_and_aggregate_data(path_to_database, layers)
    # create a mapbox plot
    #values = ["graverij klein", "graverij groot"]
    ## filter the PARAMETER_NAAM column
    #punten = punten[punten["PARAMETER_NAAM"].isin(values)]
    ## plot the data in a mapbox plot
    #fig = px.scatter_mapbox(punten, lat="lat", lon="lon", color="PARAMETER_NAAM", hover_name="WAARDE", zoom=8, height=800)
    #fig.update_layout(mapbox_style="open-street-map")
    #
    #fig.show()
    # Schadebeeld door de jaren heen
    punten['year'] = [time.year for time in punten['DATUM']]
    punten_group = punten.groupby('year')
    res_2019 = punten[punten['year'] == 2019]
    # plot the PARAMETER_NAAM column in a histogram with the year as color
    fig = px.histogram(punten, x="OBJECT_NAAM", color="year", title="Damage per year")
    fig.show()
    # plot the count of inspections per year
    fig = px.histogram(punten, x="year", title="Number of inspections per year")
    fig.write_html("../results/number_of_inspections_per_year.html")










