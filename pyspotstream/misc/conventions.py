def get_acronym(x):
    """
    Maps long name to short acronym, e.g., `"mean_absolute_error"` to `"MAE"`.

    Args:
        x (string): long name

    Returns:
        string: acronym
    """
    if x == "mean_absolute_error":
        return "MAE"
    else:
        return "Acronym unknown."
