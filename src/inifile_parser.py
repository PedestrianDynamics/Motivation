import json
from sys import argv
from typing import Dict, List, Optional, Tuple, TypeAlias

import jsonschema  # type: ignore
import shapely  # type: ignore

Point: TypeAlias = Tuple[float, float]


def parse_destinations(json_data: dict) -> Dict[int, List[List[Point]]]:
    """
    Parses the 'destinations' object from a JSON string into a Python dictionary.

    Args:
        json_data: A dict containing JSON data with a 'destinations' object.

    Returns:
        A dictionary with the parsed 'destinations' object. The dictionary maps
        IDs (int) to lists of polygons, where each polygon is a list of (x, y) tuples.
    """

    destinations = {}
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
) -> Dict[int, List[Tuple[Point, float]]]:
    """
    Parses the 'way_points' object from a JSON string into a Python dictionary.

    Args:
        json_data: A dict containing JSON data with a 'way_points' object.

    Returns:
        A dictionary with the parsed 'way_points' object. The dictionary maps
        ID (int) to lists of tuples, where each tuple contains a (x, y) point
        and a floating-point number representing a distance.
    """

    way_points = {}
    for wp_id, way_point in enumerate(json_data["way_points"]):
        wp_list = [way_point["coordinates"], way_point["distance"]]
        way_points[wp_id] = wp_list

    return way_points


def parse_distribution_polygons(
    json_data: dict,
) -> Dict[int, shapely.Polygon]:
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


def parse_jpsvis_doors(json_data: dict) -> Dict[int, List[List[float]]]:
    """
    Parses a JSON string containing information about jpsvis doors and returns a dictionary
    mapping area IDs to a list of coordinates that define the doros.

    :param json_str: A JSON string containing information about accessible areas.
    :return: A dictionary mapping area IDs to a list of coordinates that define the doors.
    """

    doors: Dict[int, List[List[float]]] = {}

    if "jpsvis_doors" in json_data:
        doors_dict = json_data["jpsvis_doors"]

        # Iterate through the accessible areas dictionary and extract the coordinates for each area
        for area_id, coordinates_list in enumerate(doors_dict):
            doors[int(area_id)] = coordinates_list["vertices"]

    return doors


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
    for area_id, coordinates_list in enumerate(areas_dict):
        areas[int(area_id)] = coordinates_list["vertices"]

    return areas


def parse_fps(json_data: dict) -> Optional[int]:
    if "fps" in json_data:
        return int(json_data["fps"])

    return None


def parse_time_step(json_data: dict) -> Optional[float]:
    if "time_step" in json_data:
        return float(json_data["time_step"])

    return None


def parse_simulation_time(json_data: dict) -> Optional[int]:
    if "simulation_time" in json_data:
        return int(json_data["simulation_time"])

    return None


def Print(obj: dict, name: str):
    print(f"{name}: ")
    for id, poly in obj.items():
        print(f"{id=}, {name=}: {poly=}")
    print("-----------")


if __name__ == "__main__":
    if len(argv) < 3:
        exit(f"usage: {argv[0]} inifile.json schema_file.json")

    inifile = argv[1]
    schema_file = argv[2]
    schema = None
    with open(schema_file, "r") as s:
        schema_str = s.read()
        schema = json.loads(schema_str)

    with open(inifile, "r") as f:
        json_str = f.read()

        try:
            data = json.loads(json_str)
            if schema:
                print("Validate json file ...\n-----------")
                jsonschema.validate(instance=data, schema=schema)

            accessible_areas = parse_accessible_areas(data)
            destinations = parse_destinations(data)
            jpsvis_doors = parse_jpsvis_doors(data)
            distribution_polygons = parse_distribution_polygons(data)

            way_points = parse_way_points(data)
            profiles = parse_velocity_model_parameter_profiles(data)
            version = data["version"]
            fps = parse_fps(data)
            time_step = parse_time_step(data)
            sim_time = parse_simulation_time(data)
            print(f"{version=}")
            print(f"{fps=}")
            print(f"{time_step=}")
            print(f"{sim_time=}")
            Print(accessible_areas, "accessible area")
            Print(destinations, "destination")
            Print(jpsvis_doors, "jpsvis_doors")
            Print(distribution_polygons, "distribution polygon")
            Print(profiles, "profile")
            Print(way_points, "way_point")

        except jsonschema.exceptions.ValidationError as e:
            print("Invalid JSON:", e)
        except json.decoder.JSONDecodeError as e:
            print("Invalid JSON syntax:", e)
        except ValueError as e:
            print("Invalid JSON:", e)
