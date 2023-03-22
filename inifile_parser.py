import json
from sys import argv
from typing import Dict, List, Tuple

from typing import Dict, List, Tuple


def parse_destinations(json_data: dict) -> Dict[int, List[List[Tuple[float, float]]]]:
    """
    Parses the 'destinations' object from a JSON string into a Python dictionary.

    Args:
        json_data: A dict containing JSON data with a 'destinations' object.

    Returns:
        A dictionary with the parsed 'destinations' object. The dictionary maps
        IDs (int) to lists of polygons, where each polygon is a list of (x, y) tuples.
    """

    destinations = {}
    for id_str, dest_list in json_data["destinations"].items():
        destinations[int(id_str)] = dest_list

    return destinations


def parse_velocity_model_parameter_profiles(json_data: dict) -> Dict[int, List[float]]:
    """
    Parses the 'velocity_model_parameter_profiles' object from a JSON string into a Python dictionary.

    Args:
        json_data: A dict containing JSON data with a 'velocity_model_parameter_profiles' object.

    Returns:
        A dictionary with the parsed 'velocity_model_parameter_profiles' object. The dictionary maps
        ID (int) to lists of floating-point numbers.
    """

    profiles = {}
    for id_str, profile_list in json_data["velocity_model_parameter_profiles"].items():
        profiles[int(id_str)] = profile_list
    return profiles


def parse_way_points(
    json_data: dict,
) -> Dict[int, List[Tuple[Tuple[float, float], float]]]:
    """
    Parses the 'way_points' object from a JSON string into a Python dictionary.

    Args:
        json_data: A dict containing JSON data with a 'way_points' object.

    Returns:
        A dictionary with the parsed 'way_points' object. The dictionary maps
        ID (int) to lists of tuples, where each tuple contains a (x, y) point
        and a floating-point number representing the point's time offset.
    """

    way_points = {}
    for id_str, wp_list in json_data["way_points"].items():
        way_points[int(id_str)] = wp_list

    return way_points


def parse_distribution_polygons(
    json_data: dict,
) -> Dict[int, List[List[Tuple[float, float]]]]:
    """
    Parses the 'distribution_polygons' object from a JSON string into a Python dictionary.

    Args:
        json_data: A dict containing JSON data with a 'distribution_polygons' object.

    Returns:
        A dictionary with the parsed 'distribution_polygons' object. The dictionary maps
        ID (int) to lists of polygons, where each polygon is a list of (x, y) tuples.
    """
    data = json.loads(json_str)
    distribution_polygons = {}
    for id_str, polygon in data["distribution_polygons"].items():
        distribution_polygons[int(id_str)] = polygon
    return distribution_polygons


def parse_accessible_areas(json_data: dict) -> Dict[int, List[List[float]]]:
    """
    Parses a JSON string containing information about accessible areas and returns a dictionary
    mapping area IDs to a list of coordinates that define the area.

    :param json_str: A JSON string containing information about accessible areas.
    :return: A dictionary mapping area IDs to a list of coordinates that define the area.
    """
    areas = {}

    areas_dict = json_data["accessible_areas"]

    # Iterate through the accessible areas dictionary and extract the coordinates for each area
    for area_id, coordinates_list in areas_dict.items():
        areas[int(area_id)] = coordinates_list

    return areas


def parse_fps(json_data: dict) -> float | None:
    if "fps" in json_data:
        return json_data["fps"]

    return None


def parse_dt(json_data: dict) -> float | None:
    if "dt" in json_data:
        return json_data["dt"]

    return None


def Print(obj: dict, name: str):
    for id, poly in obj.items():
        print(f"id: {id}, {name}: {poly}")
    print("-----------")


if __name__ == "__main__":
    if len(argv) < 2:
        exit(f"usage: {argv[0]} inifile.json")

    inifile = argv[1]
    with open(inifile, "r") as f:
        json_str = f.read()
        data = json.loads(json_str)
        accessible_areas = parse_accessible_areas(data)
        destinations = parse_destinations(data)
        distribution_polygons = parse_distribution_polygons(data)
        way_points = parse_way_points(data)
        profiles = parse_velocity_model_parameter_profiles(data)
        fps = parse_fps(data)
        dt = parse_dt(data)

        print(f"fps: {fps}")
        print(f"dt: {dt}")
        Print(accessible_areas, "accessible area")
        Print(destinations, "destination")
        Print(distribution_polygons, "distribution polygon")
        Print(profiles, "profile")
        Print(way_points, "way_point")
