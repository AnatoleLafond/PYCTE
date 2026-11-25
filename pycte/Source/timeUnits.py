"""Time units (timeUnit keyword) and conversion to seconds; an unknown unit is an error."""

SECONDS = {
    's': 1.0, 'sec': 1.0, 'second': 1.0, 'seconds': 1.0,
    'm': 60.0, 'min': 60.0, 'minute': 60.0, 'minutes': 60.0,
    'h': 3600.0, 'hr': 3600.0, 'hour': 3600.0, 'hours': 3600.0,
    'd': 86400.0, 'day': 86400.0, 'days': 86400.0,
    'y': 31557600.0, 'yr': 31557600.0, 'a': 31557600.0, 'year': 31557600.0, 'years': 31557600.0,
}

_SHORT = {1.0: 's', 60.0: 'min', 3600.0: 'h', 86400.0: 'd', 31557600.0: 'y'}


def secondsPerUnit(unit):
    """Seconds in one `unit` (year = 365.25 days); ValueError if unknown."""
    factor = SECONDS.get(str(unit).strip().lower())
    if factor is None:
        raise ValueError(f"pycte : unknown timeUnit {unit!r} (use one of {', '.join(SECONDS)})")
    return factor


def toSeconds(value, unit):
    """`value` expressed in `unit` -> seconds."""
    return value * secondsPerUnit(unit)


def shortName(unit):
    """Short name of `unit`: 's', 'min', 'h', 'd' or 'y'."""
    return _SHORT[secondsPerUnit(unit)]
