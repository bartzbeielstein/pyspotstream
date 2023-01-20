from requests import get


def download_from_url(url, filename):
    """
    Download file from URL.

    Args:
        url (str): URL
        filename (str): file name
    """
    req = get(url)
    with open(filename, "wb") as f:
        f.write(req.content)
