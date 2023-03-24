import json
import xml.etree.ElementTree as ET
from sys import argv

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def add_transitions(root: ET.Element, data: dict) -> None:
    """Add transition to xml file

    :param root:
    :type root: ET.Element
    :param data:
    :type data: dict
    :returns:

    """
    if not "jpsvis_doors" in data:
        return

    destinations = data["jpsvis_doors"]
    transitions = ET.SubElement(root, "transitions")

    for dest in destinations:
        t = ET.SubElement(
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
        v1, v2 = dest["vertices"]
        ET.SubElement(t, "vertex", {"px": str(v1[0]), "py": str(v1[1])})
        ET.SubElement(t, "vertex", {"px": str(v2[0]), "py": str(v2[1])})


def add_room(root: ET.Element, data: dict) -> None:
    """Add room to xml file.

    :param root:
    :type root: ET.Element
    :param data:
    :type data: dict
    :returns:

    """
    rooms = ET.SubElement(root, "rooms")
    polygons = [Polygon(p["vertices"]) for p in data["accessible_areas"]]
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
    if len(argv) < 2:
        exit(f"usage: {argv[0]} inifile.json")

    json_file = argv[1]
    print(f"<< {json_file}")
    data = None
    with open(json_file, "r") as f:
        data = json.load(f)

    root = ET.Element("geometry")
    root.set("version", "0.8")
    root.set("caption", "experiment")
    root.set("unit", "m")
    add_room(root, data)
    add_transitions(root, data)
    tree = ET.ElementTree(root)
    # xml_file = json_file.split(".json")[0] + ".xml"
    xml_file = "geometry.xml"
    print(f">> {xml_file}")
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
