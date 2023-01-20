import pandas as pd
from pyspotstream.data.download import download_from_url
import numpy as np
from datetime import datetime
import time
import re
import os


def get_lat_lon(x):
    if x != "Unknown":
        xyst = re.findall(r'\((.+)\)', x)
        xy = xyst[0].split(" ")
        lat = float(xy[0])
        lon = float(xy[1])
        return [lat, lon]
    else:
        return [-1.0, -1.0]


def get_opm(opm_url="https://data.ct.gov/api/views/5mzw-sjtu/rows.csv?accessType=DOWNLOAD",
            filename="opm_data.csv",
            overwrite=False):
    if not os.path.isfile(filename) or overwrite:
        print('Downloading to: ' + filename)
        try:
            download_from_url(url=opm_url, filename=filename)
            print("Finished.")
        except Exception as err:
            print(err)
            print('Could not download.')
    else:
        print("File " + filename + " already exists. To enforce download, set 'overwrite=True'.")


def load_opm(filename="opm_data.csv",
             data_type="num",
             n=None,
             sorted=True,
             verbose=False):
    # df = pd.read_pickle("OPM.pkl")
    dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y')
    df = pd.read_csv(filename, date_parser=dateparse)
    v = "Date Recorded"
    index = df[v].isnull()
    df.dropna(subset=[v], inplace=True)
    df.reset_index(inplace=True)
    df = df.assign(dti_rec=pd.to_datetime(df[v], infer_datetime_format=True))
    df = df.assign(timestamp_rec=list(map(lambda x: int(time.mktime(x.timetuple())), df["dti_rec"])))
    t0 = pd.to_datetime("2001-09-30", infer_datetime_format=True)
    index = df["dti_rec"] < t0
    df = df.drop(df[df["dti_rec"] < t0].index)
    v = "Town"
    df = df.assign(town_hash=list(map(lambda x: str(hash(x)), df[v])))
    v = "town_hash"
    v = "Address"
    index = df[v].isnull()
    df.loc[index, v] = "Unknown"
    df = df.assign(address_hash=list(map(lambda x: str(hash(x)), df[v])))
    v = "Assessed Value"
    min_val = 2000
    index = df[v] < min_val
    df = df.drop(df[df[v] < min_val].index)
    max_val = 1e8
    index = df[v] > max_val
    df = df.drop(df[df["Assessed Value"] == 138958820.0].index)
    v = "Sale Amount"
    index = df[v] < min_val
    df = df.drop(df[df[v] < min_val].index)
    v = "Sale Amount"
    max_val = 2e8
    index = df[v] > max_val
    df = df.drop(df[df[v] > max_val].index)
    v = "Sales Ratio"
    index = (df[v] < 1e-4)
    df.loc[index, "Assessed Value"] = df.loc[index, "Assessed Value"] * 1e4
    df.loc[index, "Sales Ratio"] = df.loc[index, "Assessed Value"] / df.loc[index, "Sale Amount"]
    index = (df[v] < 1e-4)
    df.loc[index, "Assessed Value"] - df.loc[index, "Sale Amount"]
    v = "Assessed Value"
    index = (df[v] > 1e8)
    df = df.drop(df[df[v] > 1e8].index)
    v = "Assessed Value"
    v = "Property Type"
    index = df[v].isnull()
    df.loc[index, v] = "Unknown"
    v = "Residential Type"
    index = df[v].isnull()
    df.loc[index, v] = "Unknown"
    v = "Non Use Code"
    index = df[v].isnull()
    df.loc[index, v] = "Unknown"
    v = "Assessor Remarks"
    index = df[v].isnull()
    df.loc[index, v] = "Unknown"
    v = "OPM remarks"
    index = df[v].isnull()
    df.loc[index, v] = "Unknown"
    v = "Location"
    index = df[v].isnull()
    df.loc[index, v] = "Unknown"
    df = df.assign(lat=list(map(lambda x: get_lat_lon(x)[0], df[v])))
    df = df.assign(lon=list(map(lambda x: get_lat_lon(x)[1], df[v])))
    df = df.reset_index(drop=True)
    for c, dtype in zip(df.columns, df.dtypes):
        if dtype == np.float64:
            df[c] = df[c].astype(np.float32)
        if dtype == np.int64:
            df[c] = df[c].astype(np.int32)
    # Sort
    if sorted:
        df = df.sort_values(by=["timestamp_rec"], ignore_index=True)

    num_cols = [
        "List Year",  # Bekanntmachung (Jahr) als integer
        "Assessed Value",
        "Sale Amount",
        "Sales Ratio",
        "timestamp_rec",  # Verkaufsdatum (Tag) als integer
        "lat",
        "lon",
    ]
    cat_cols = [
        "Town",
        "Address",
        "Property Type",
        "Residential Type",
        "Non Use Code",
        "Assessor Remarks",
        "OPM remarks",
    ]

    num_cat_cols = num_cols + cat_cols
    # Wie cal_cols, nur teilweise Hash-Werte:
    cat_cols_hashed = [
        "town_hash",
        "address_hash",
        "Property Type",
        "Residential Type",
        "Non Use Code",
        "Assessor Remarks",
        "OPM remarks",
    ]

    # Datum infos in unterschiedlichen Formaten:
    date_cols = ["Date Recorded", "timestamp_rec", "dti_rec"]

    match data_type:
        case "num":
            x = df[num_cols]
        case "cat":
            x = df[cat_cols]
        case "num_cat":
            x = df[num_cat_cols]
        case _:
            print("Wrong data type. Should be one of 'num', 'cat' or 'num_cat'.")

    # Select sample size
    if n is None:
        n = df.shape[0]
    x = x.iloc[range(n), :]

    # drop and return "Sale Amount"
    y = pd.DataFrame(x.pop("Sale Amount"))

    if verbose:
        print(x.describe(include="all"))
        print(y.describe(include="all"))
    return x, y
