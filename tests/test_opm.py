import pandas as pd

from sklearn.utils import Bunch
from pyspotstream.datasets import fetch_opm


def test_Xy():
    for incn in [True, False]:
        for incc in [True, False]:
            if not incn and not incc:
                continue
            X, y = fetch_opm(return_X_y=True, include_numeric=incn, include_categorical=incc)
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)

            columns = list(X.columns)
            for c in ["List Year", "Assessed Value", "Sale Amount", "Sales Ratio", "lat", "lon"]:
                if incn:
                    assert c in columns
                else:
                    assert c not in columns
            for c in [
                "Town",
                "Address",
                "Property Type",
                "Residential Type",
                "Non Use Code",
                "Assessor Remarks",
                "OPM remarks",
            ]:
                if incc:
                    assert c in columns
                else:
                    assert c not in columns


def test_bunch():
    bunch = fetch_opm(return_X_y=False)
    assert isinstance(bunch, Bunch)
