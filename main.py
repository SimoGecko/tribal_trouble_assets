import xml.etree.ElementTree as ET
import json
import struct
import base64

# Load XML
tree = ET.parse("mesh.xml")
root = tree.getroot()

positions = []
normals = []
colors = []
uvs = []
indices = []

vertex_map = {}
current_index = 0

# Go through polygons
for polygon in root.find("polygons").findall("polygon"):
    poly_indices = []
    for vertex in polygon.findall("vertex"):
        # Collect vertex attributes
        x, y, z = float(vertex.attrib["x"]), float(vertex.attrib["y"]), float(vertex.attrib["z"])
        nx, ny, nz = float(vertex.attrib["nx"]), float(vertex.attrib["ny"]), float(vertex.attrib["nz"])
        r, g, b, a = float(vertex.attrib["r"]), float(vertex.attrib["g"]), float(vertex.attrib["b"]), float(vertex.attrib["a"])
        u, v = float(vertex.attrib["u"]), float(vertex.attrib["v"])
        
        key = (x, y, z, nx, ny, nz, r, g, b, a, u, v)
        if key not in vertex_map:
            vertex_map[key] = current_index
            current_index += 1
            positions.extend([x, y, z])
            normals.extend([nx, ny, nz])
            colors.extend([r, g, b, a])
            uvs.extend([u, v])
        
        poly_indices.append(vertex_map[key])
    
    # Assuming all polygons are triangles
    if len(poly_indices) == 3:
        indices.extend(poly_indices)
    elif len(poly_indices) > 3:
        # Triangulate simple convex polygon (fan method)
        for i in range(1, len(poly_indices)-1):
            indices.extend([poly_indices[0], poly_indices[i], poly_indices[i+1]])

# Helper to convert float lists to base64
def float_to_b64(floats):
    b = struct.pack("<" + "f"*len(floats), *floats)
    return base64.b64encode(b).decode('ascii')

def uint16_to_b64(uints):
    b = struct.pack("<" + "H"*len(uints), *uints)
    return base64.b64encode(b).decode('ascii')

# Build minimal glTF
gltf = {
    "asset": {"version": "2.0"},
    "buffers": [],
    "bufferViews": [],
    "accessors": [],
    "meshes": [
        {
            "primitives": [
                {
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1,
                        "COLOR_0": 2,
                        "TEXCOORD_0": 3
                    },
                    "indices": 4
                }
            ]
        }
    ],
    "scenes": [{"nodes": [0]}],
    "nodes": [{"mesh": 0}]
}

# Create buffer data
position_b64 = float_to_b64(positions)
normal_b64 = float_to_b64(normals)
color_b64 = float_to_b64(colors)
uv_b64 = float_to_b64(uvs)
index_b64 = uint16_to_b64(indices)

# Each bufferView references data inside a buffer (all inline here)
buffers_data = [
    position_b64, normal_b64, color_b64, uv_b64, index_b64
]

bufferViews = []
accessors = []

byte_offset = 0
for i, data in enumerate(buffers_data):
    bufferViews.append({
        "buffer": 0,
        "byteOffset": byte_offset,
        "byteLength": len(base64.b64decode(data))
    })
    byte_offset += len(base64.b64decode(data))

# Accessors (simple)
accessors = [
    {"bufferView": 0, "componentType": 5126, "count": len(positions)//3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5126, "count": len(normals)//3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5126, "count": len(colors)//4, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5126, "count": len(uvs)//2, "type": "VEC2"},
    {"bufferView": 4, "componentType": 5123, "count": len(indices), "type": "SCALAR"}
]

gltf["buffers"].append({
    "uri": "data:application/octet-stream;base64," + "".join(buffers_data),
    "byteLength": byte_offset
})
gltf["bufferViews"] = bufferViews
gltf["accessors"] = accessors

# Save glTF
with open("mesh.gltf", "w") as f:
    json.dump(gltf, f, indent=2)

print("Conversion done!")
