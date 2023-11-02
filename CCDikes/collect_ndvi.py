import datetime
import pystac_client
import planetary_computer
import stackstac
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

BAND = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B08": "nir",
    "B09": "evaporation",
    "B11": "swir1",
    "SCL": "SCL",
}


def get_ndvi_on_date_bbox(bbox, date, inspection_points):
    # get date
    sub_date = date
    last_month = date - datetime.timedelta(days=60)
    datetime_string = f"{last_month.year}-{last_month.strftime('%m')}-{last_month.strftime('%d')}/{sub_date.year}-{sub_date.strftime('%m')}-{sub_date.strftime('%d')}"
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=[
            "sentinel-2-l2a"
        ],  # atmospherically corrected Surface Reflectances (SR)
        bbox=bbox,
        max_items=20,
        datetime=datetime_string,  # "2020-01-01/2020-01-31",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = search.item_collection()
    print(f"{len(items)} items found in catalog search.")
    if len(items) > 0:
        data = stackstac.stack(
            items,
            epsg=4326,  # if you want to cast the data to a common crs
            assets=list(BAND.keys()),
            bounds_latlon=bbox.tolist(),
            sortby_date="desc",  # maybe you want to sort the data by date?
        )
        # calculate the median
        median = data.median(dim="time").compute()
        #median = data.chunk({'time': -1}).quantile(q=0.8, dim="time").compute()
        # calculate the Normalised difference moisture index
        red = median.sel({"band": "B04"})
        nir = median.sel({"band": "B08"})
        ndvi = (nir - red) / (red + nir)  # this is still a lazy Dask computation
        #ndvi = ndvi.compute()  # this will actually run the computation
        # sample the data at the inspection points
        x = xr.DataArray(list(inspection_points[0, :]), dims='z')
        y = xr.DataArray(list(inspection_points[1, :]), dims='z')
        time_ndvi = ndvi.sel(x=x, y=y, method='nearest').to_dataframe(name="ndvi").reset_index()
        return time_ndvi
    else:
        return None


if __name__ == "__main__":
    path_to_points = "test_cross_sections.csv"
    points = pd.read_csv(path_to_points)
    # to geopandas
    points = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.x, points.y))
    # set crs
    points.crs = "EPSG:28992"
    # # transform to latlon
    points = points.to_crs("EPSG:4326")
    bbox = points.total_bounds
    # 1rst of september 2023
    date_now = datetime.datetime(2022, 9, 1)
    inspection_points = np.array([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])
    ndvi = get_ndvi_on_date_bbox(bbox, date_now, inspection_points)
    ndvi.to_csv("../data/test_data_ndvi_summer.csv")
    # path_to_database = "..\data\Digispectie2017_2022\Digispectie2017_2022.gdb"
    # layers = ['Inspectie2017_najaar_punt',
    #           'Inspectie2018_najaar_punt',
    #           'Inspectie2019_najaar_punt',
    #           'Inspectie2020_voorjaar_punt',
    #           'Inspectie2020_najaar_punt',
    #           'Inspectie2021_voorjaar_punt',
    #           'Inspectie2021_najaar_punt',
    #           'Inspectie2022_voorjaar_punt']
    # punten = []
    # for layer in layers:
    #     if "punt" in layer:
    #         punten.append(gpd.read_file(path_to_database, layer=layer))
    # punten = gpd.GeoDataFrame(pd.concat(punten, axis=0, sort=False))
    # # set crs
    # punten.crs = "EPSG:28992"
    # # transform to latlon
    # punten = punten.to_crs("EPSG:4326")
    # # get x and y as columns
    # punten["x"] = punten.geometry.x
    # punten["y"] = punten.geometry.y
    # # get points as array [:,2]
    # inspection_points = punten[["x", "y"]].to_numpy()
    # bbox = punten.total_bounds
    # date_now = datetime.datetime.now()
    # ndvi = get_ndvi_on_date_bbox(bbox, date_now, inspection_points)
    # # save to csv
    # ndvi.to_csv("../data/ndvi.csv")
