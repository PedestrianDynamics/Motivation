import json
from sys import argv
from typing import Dict, List, Tuple

from typing import Dict, List, Tuple
import shapely


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
    print(json_data["destinations"])
    for destination in json_data["destinations"]:
        id_str = destination["id"]
        dest_list = destination["vertices"]
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
    for profile in json_data["velocity_model_parameter_profiles"]:
        id_str = profile["id"]
        time_gap = profile["time_gap"]
        tau = profile["tau"]
        v0 = profile["v0"]
        radius = profile["radius"]
        profiles[int(id_str)] = [time_gap, tau, v0, radius]
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
    for wp_id, way_point in enumerate(json_data["way_points"]):
        wp_list = [way_point["coordinates"], way_point["distance"]]
        way_points[wp_id] = wp_list

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
    distribution_polygons = {}
    for id_polygon in json_data["distribution_polygons"]:
        id_str = id_polygon["id"]
        polygon = id_polygon["vertices"]
        distribution_polygons[int(id_str)] = shapely.Polygon(polygon)
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
    print(areas_dict)
    for area_id, coordinates_list in enumerate(areas_dict):
        areas[int(area_id)] = coordinates_list["vertices"]

    return areas


def parse_fps(json_data: dict) -> float | None:
    if "fps" in json_data:
        return json_data["fps"]

    return None


def parse_time_step(json_data: dict) -> float | None:
    if "time_step" in json_data:
        return json_data["time_step"]

    return None


def parse_simulation_time(json_data: dict) -> float | None:
    if "simulation_time" in json_data:
        return json_data["simulation_time"]

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
        time_step = parse_time_step(data)
        sim_time = parse_simulation_time(data)
        print(f"fps: {fps}")
        print(f"time_step: {time_step}")
        print(f"simulation time: {sim_time}")
        Print(accessible_areas, "accessible area")
        Print(destinations, "destination")
        Print(distribution_polygons, "distribution polygon")
        Print(profiles, "profile")
        Print(way_points, "way_point")
