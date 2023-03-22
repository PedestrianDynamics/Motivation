import xml.etree.ElementTree as ET
import json
from sys import argv

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
rooms = ET.SubElement(root, "rooms")

# create a sub-element for each room
for room_id, points in data["accessible_areas"].items():
    room = ET.SubElement(rooms, "room")
    room.set("id", room_id)
    room.set("caption", "room")

    # create a sub-element for the subroom
    subroom = ET.SubElement(room, "subroom")
    subroom.set("id", "0")
    subroom.set("caption", "subroom")
    subroom.set("class", "subroom")

    # create a polygon element for each set of points
    polygon = ET.SubElement(subroom, "polygon")
    polygon.set("caption", "wall")
    polygon.set("type", "internal")
    for i, point in enumerate(points):
        vertex = ET.SubElement(polygon, "vertex")
        vertex.set("px", str(point[0]))
        vertex.set("py", str(point[1]))

    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", str(points[0][0]))
    vertex.set("py", str(points[0][1]))

tree = ET.ElementTree(root)
xml_file = json_file.split(".json")[0] + ".xml"
print(f">> {xml_file}")
tree.write(xml_file, encoding="utf-8", xml_declaration=True)
