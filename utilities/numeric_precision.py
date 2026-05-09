import json
from typing import Any


MAX_DECIMAL_PLACES = 3


def cap_float_precision(value: float, max_decimal_places: int = MAX_DECIMAL_PLACES) -> float:
    """Round a float to the configured global decimal cap."""
    return round(float(value), int(max_decimal_places))


def format_decimal(value: float, max_decimal_places: int = MAX_DECIMAL_PLACES) -> str:
    """Format a number as text with a fixed global decimal precision."""
    return f"{cap_float_precision(value, max_decimal_places):.{int(max_decimal_places)}f}"


def cap_numeric_precision(payload: Any, max_decimal_places: int = MAX_DECIMAL_PLACES) -> Any:
    """Recursively round all float values in nested payloads."""
    if isinstance(payload, float):
        return cap_float_precision(payload, max_decimal_places)
    if isinstance(payload, list):
        return [cap_numeric_precision(item, max_decimal_places) for item in payload]
    if isinstance(payload, tuple):
        return tuple(cap_numeric_precision(item, max_decimal_places) for item in payload)
    if isinstance(payload, dict):
        return {
            key: cap_numeric_precision(value, max_decimal_places)
            for key, value in payload.items()
        }
    return payload


def dumps_capped(payload: Any, **json_kwargs: Any) -> str:
    """Serialize payload to JSON after capping float precision globally."""
    return json.dumps(cap_numeric_precision(payload), **json_kwargs)
