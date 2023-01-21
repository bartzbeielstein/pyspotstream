import hashlib
from pathlib import Path

def sha256sum(filename):
    hash = hashlib.sha256()
    buf = bytearray(1024 * 1024) # Reusable 1MB buffer
    view = memoryview(buf)
    with open(filename, "rb") as fd:
        while True:
            size = fd.readinto(buf)
            if size == 0:
                break  # EOF
            hash.update(view[:size])
        return hash.hexdigest()

def get_data_home(data_home=None) -> str:
    """Return the path of the pyspotstream data directory.
    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data directory is set to a folder named 'pyspotstream_data' in the
    user home folder.
    Alternatively, it can be set by the 'PYSPOTSTREAM_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    
    Parameters
    ----------
    data_home : str, default=None
        The path to pyspotstream data directory. If `None`, the default path
        is `~/pyspotstream_data`.
        
    Returns
    -------
    data_home: str
        The path to the pyspotstream data directory.
    """
    if data_home is None:
        data_home = environ.get("PYSPOTSTREAM_DATA", 
                                Path.home() / "pyspotstream_data")
    # Ensure data_home is a Path() object pointing to an absolute path
    data_home = Path(data_home).absolute()
    # Create data directory if it does not exists.
    data_home.mkdir(parents=True, exist_ok=True)
    
    return data_home

