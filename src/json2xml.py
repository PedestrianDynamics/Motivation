"""
Parse geometry infos from json file and create
a <geometry.xml> file

Looking in json for
- jpsvis_doors
- accessible_areas: Make a union on all walls to get a nice polygon

"""

import json
import sys
import xml.etree.ElementTree as ET
from typing import Any, Dict

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def add_transitions(_root: ET.Element, _data: Dict[str, Any]) -> None:
    """Add transition to xml file

    :param root:
    :type root: ET.Element
    :param data:
    :type data: dict
    :returns:

    """
    if "jpsvis_doors" not in _data:
        return

    destinations = _data["jpsvis_doors"]
    transitions = ET.SubElement(_root, "transitions")

    for dest in destinations:
        transition = ET.SubElement(
            transitions,
            "transition",
            {
                "id": str(dest["id"]),
                "caption": "",
                "type": "emergency",
                "room1_id": "1",
                "subroom1_id": "0",
                "room2_id": "-1",
                "subroom2_id": "-1",
            },
        )
        vertice_1, vertice_2 = dest["vertices"]
        ET.SubElement(
            transition, "vertex", {"px": str(vertice_1[0]), "py": str(vertice_1[1])}
        )
        ET.SubElement(
            transition, "vertex", {"px": str(vertice_2[0]), "py": str(vertice_2[1])}
        )


def add_room(_root: ET.Element, _data: Dict[str, Any]) -> None:
    """Add room to xml file.

    :param root:
    :type root: ET.Element
    :param data:
    :type data: dict
    :returns:

    """
    rooms = ET.SubElement(_root, "rooms")
    polygons = [Polygon(p["vertices"]) for p in _data["accessible_areas"]]
    multi_poly = MultiPolygon(polygons)
    merged_poly = unary_union(multi_poly)
    room_id = 1
    points = merged_poly.exterior.coords
    room = ET.SubElement(rooms, "room")
    room.set("id", str(room_id))
    room.set("caption", "room")

    subroom = ET.SubElement(room, "subroom")
    subroom.set("id", "0")
    subroom.set("caption", "subroom")
    subroom.set("class", "subroom")

    polygon = ET.SubElement(subroom, "polygon")
    polygon.set("caption", "wall")
    polygon.set("type", "internal")

    for _, point in enumerate(points):
        vertex = ET.SubElement(polygon, "vertex")
        vertex.set("px", str(point[0]))
        vertex.set("py", str(point[1]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} inifile.json")

    JSON_FILE = sys.argv[1]
    print(f"<< {JSON_FILE}")
    DATA = None
    with open(JSON_FILE, "r", encoding="utf8") as f:
        DATA = json.load(f)

    root = ET.Element("geometry")
    root.set("version", "0.8")
    root.set("caption", "experiment")
    root.set("unit", "m")
    add_room(root, DATA)
    add_transitions(root, DATA)
    tree = ET.ElementTree(root)
    # xml_file = json_file.split(".json")[0] + ".xml"
    XML_FILE = "geometry.xml"
    print(f">> {XML_FILE}")
    tree.write(XML_FILE, encoding="utf-8", xml_declaration=True)
