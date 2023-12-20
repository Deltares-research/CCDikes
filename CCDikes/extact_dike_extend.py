import requests
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString
import numpy as np
import plotly.graph_objects as go
import pyproj
import math
from shapely.ops import substring





def plot_cross_sections_on_mapbox(point_list, epsg_in=28992, epsg_out=4326):
    """
    Create a Mapbox plot in Plotly with points from cross sections.

    Parameters:
    - cross_sections (list of LineString): List of LineStrings representing cross sections.
    - epsg_in (int): Input EPSG code (source coordinate system).
    - epsg_out (int): Output EPSG code (destination coordinate system).

    Returns:
    - Plotly Figure: Mapbox plot with cross section points.
    """

    # Initialize a projection transformer to convert coordinates
    transformer = pyproj.Transformer.from_crs(epsg_in, epsg_out, always_xy=True)

    # Initialize an empty list to store coordinates
    lon_coords = []
    lat_coords = []



    # Convert coordinates from EPSG:28992 to EPSG:4326
    transformed_coords = [transformer.transform(x, y) for x, y in point_list]

    # Split the transformed coordinates into lon and lat
    lon, lat = zip(*transformed_coords)

    # Append the converted coordinates to the lists
    lon_coords.extend(lon)
    lat_coords.extend(lat)

    # Create a Mapbox scatter plot
    fig = go.Figure(go.Scattermapbox(
        mode="markers",
        lon=lon_coords,
        lat=lat_coords,
        marker=go.scattermapbox.Marker(
            size=8,
            color="blue",
        ),
    ))

    # Set up Mapbox layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=11,
    )

    return fig


def get_perpendicular_cross_section_to_trajectory(dijkvak_traj_sub: LineString, length_cross_section: float) -> LineString:
    """
    Return a perpendicular cross-section to the trajectory at the specified distance from the start of the trajectory.

    :param dijkvak_traj_sub: trajectory
    :param length_cross_section: length of the cross-section in meters

    :return: LineString
    """

    left = dijkvak_traj_sub.parallel_offset(length_cross_section / 2, 'left')
    right = dijkvak_traj_sub.parallel_offset(length_cross_section / 2, 'right')
    try:
        c = left.boundary.geoms[1]
        d = right.boundary.geoms[0]
    except IndexError:
        print('IndexError')
    cross_section_ls = LineString([c, d])
    return cross_section_ls


def get_cross_section_char_points(trajectory, distance_from_start: float, length_cross_section: float) -> list:
        """
        Return a list of xy coordinates of the  three characteristic representing a cross-section: the left point, the
        middle point and the right point. The middle point is located on the trace at the specified distance from the
        start of the trace. The left and right points are located on the perpendicular cross-section to the trace at
        the specified distance from the start of the trace.

        :param distance_from_start: distance in meters from the start of the trace for which the cross-section is generated.
        :param length_cross_section: length of the cross-section in meters.

        :return:
        """
        trajectory_to_cs_position = substring(trajectory, 0, distance_from_start)

        cross_section_linestring = get_perpendicular_cross_section_to_trajectory(trajectory_to_cs_position,
                                                                                 length_cross_section=length_cross_section)
        # Exytreme left and right points of the cross-section
        right_left_points = [(x, y) for x, y in cross_section_linestring.coords]

        # This is the middle point of the cross-section located on the trace
        x_mid, y_mid = trajectory_to_cs_position.boundary.geoms[1].x, trajectory_to_cs_position.boundary.geoms[1].y

        return [right_left_points[0], (x_mid, y_mid), right_left_points[1]]


def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))


def extract_cross_sections(trajectory, n, y):
    cross_sections = []

    for linestring in trajectory.geoms:
        # plot the trajectory
        total_distance = 0
        # every n meters, generate a cross section
        distances = np.arange(0, linestring.length, n)
        distances = distances[1:-1]
        for distance in distances:
            # Generate points along the cross section
            cross_section_points = []
            extreme_points = get_cross_section_char_points(linestring, distance, y)
            # populate between the points with equidistant points
            cross_section_points += list(getEquidistantPoints(extreme_points[0], extreme_points[1], 10))
            cross_section_points += list(getEquidistantPoints(extreme_points[1], extreme_points[2], 10))
            # plot the cross section
            total_distance -= n
            cross_sections.append(cross_section_points)
    return cross_sections


# connect to the API with WFS protocol
# this url should be in code https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?service=WFS&request=GetFeature&typeName=bestuurlijkegebieden:Provinciegebied&sql_filter=code=30&version=2.0.0
url = "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0"
dictionary = {
    'service': 'WFS',
    'version': '2.0.0',
    'request': 'GetFeature',
    'otputFormat': 'application/json',
    'typeName': 'bestuurlijkegebieden:Provinciegebied',
    'sql_filter': 'code=30'
}

# get the data
response = requests.get(url, params=dictionary)
print(response.url)
# save the data to shapefile from the response
with open('Provinciegebied_30.geojson', 'wb') as f:
    f.write(response.content)
# open the file with geopandas
from geopandas import read_file
data = read_file('Provinciegebied_30.geojson')
# keep code is 30
data = data[data['code'] == '30']
# get the wkt of the geometry
wkt_poly = data['geometry'].array[0]

# request https://waterveiligheidsportaal.nl/geoserver/nbpw/ows/wfs?srsname=EPSG:28992&typename=nbpw:dijktrajecten&version=1.1.0&request=GetFeature&outputFormat=csv
url = "https://waterveiligheidsportaal.nl/geoserver/nbpw/ows/wfs"
dictionary = {
    'srsname': 'EPSG:28992',
    'typename': 'nbpw:dijktrajecten',
    'version': '1.1.0',
    'request': 'GetFeature',
    'outputFormat': 'csv'
}
# get the data
response = requests.get(url, params=dictionary)
print(response.url)
# save the data to shapefile from the response
with open('dijktrajecten.csv', 'wb') as f:
    f.write(response.content)
# open the file with geopandas
from geopandas import read_file
data = read_file('dijktrajecten.csv')
# set the geometry
from shapely import wkt
data['geometry'] = data['geom'].apply(wkt.loads)
# set the crs
data.crs = 'EPSG:28992'

cross_sections = []
# intersect the data with the province
for index, row in data.iterrows():
    polygon_1 = row['geometry']
    if polygon_1.intersects(wkt_poly):
        cross_section = extract_cross_sections(polygon_1, 1500, 3)
        cross_sections += cross_section
# flatten the list to points and create dataframe
cross_sections = [item for sublist in cross_sections for item in sublist]
fig = plot_cross_sections_on_mapbox(cross_sections, epsg_in=28992, epsg_out=4326)
fig.show()
# save the data to csv
import pandas as pd
df = pd.DataFrame(cross_sections, columns=['x', 'y'])
df.to_csv('test_cross_sections.csv', index=False)








