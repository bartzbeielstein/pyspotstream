import pyspotstream.data.OPM as opm

# Download data, throws exception on error.
def test_download():
    filename = opm.get_opm()
    assert filename.is_file()

