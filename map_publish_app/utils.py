def parse_latlng_pair(param_str):
    """文字列 'lat,lng' を (float(lat), float(lng)) に変換する"""
    try:
        lat_str, lng_str = param_str.split(",")
        return float(lat_str.strip()), float(lng_str.strip())
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid lat/lng format: '{param_str}'")
