import xml.etree.ElementTree as ET
import json
import struct
import base64
import math

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

def rotate(x,y,z):
    # -90 deg X rotation
    return x, z, -y

# Go through polygons
for polygon in root.find("polygons").findall("polygon"):
    poly_indices = []
    for vertex in polygon.findall("vertex"):
        # Collect vertex attributes
        x, y, z = float(vertex.attrib["x"]), float(vertex.attrib["y"]), float(vertex.attrib["z"])
        nx, ny, nz = float(vertex.attrib["nx"]), float(vertex.attrib["ny"]), float(vertex.attrib["nz"])
        r, g, b, a = float(vertex.attrib["r"]), float(vertex.attrib["g"]), float(vertex.attrib["b"]), float(vertex.attrib["a"])
        u, v = float(vertex.attrib["u"]), float(vertex.attrib["v"])
        
        # Flip V
        v = 1.0 - v

        # Rotate
        x, y, z = rotate(x, y, z)
        nx, ny, nz = rotate(nx, ny, nz)

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
        print('WARNING: not a triangle - triangulating')
        # Triangulate simple convex polygon (fan method)
        for i in range(1, len(poly_indices)-1):
            indices.extend([poly_indices[0], poly_indices[i], poly_indices[i+1]])

# normalize normals
for i in range(0, len(normals), 3):
    nx, ny, nz = normals[i], normals[i+1], normals[i+2]
    length = math.sqrt(nx**2 + ny**2 + nz**2)
    if length > 0:
        normals[i]   = nx / length
        normals[i+1] = ny / length
        normals[i+2] = nz / length

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
    "nodes": [{
        "name": "node0",
        "mesh": 0,
        #"rotation": [-0.7071068, 0, 0, 0.7071068]  # quaternion for -90° X
    }]
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
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

bufferViews[0]["target"] = ARRAY_BUFFER
bufferViews[1]["target"] = ARRAY_BUFFER
bufferViews[2]["target"] = ARRAY_BUFFER
bufferViews[3]["target"] = ARRAY_BUFFER
bufferViews[4]["target"] = ELEMENT_ARRAY_BUFFER

'''
bufferViews = [
    {"buffer":0, "byteOffset":0, "byteLength":len(base64.b64decode(position_b64)), "target": ARRAY_BUFFER},
    {"buffer":0, "byteOffset":len(base64.b64decode(position_b64)), "byteLength":len(base64.b64decode(normal_b64)), "target": ARRAY_BUFFER},
    {"buffer":0, "byteOffset":..., "byteLength":..., "target": ARRAY_BUFFER},  # for COLOR_0
    {"buffer":0, "byteOffset":..., "byteLength":..., "target": ARRAY_BUFFER},  # for TEXCOORD_0
    {"buffer":0, "byteOffset":..., "byteLength":len(base64.b64decode(index_b64)), "target": ELEMENT_ARRAY_BUFFER},
]
'''

# Accessors (simple)
accessors = [
    {"bufferView": 0, "componentType": 5126, "count": len(positions)//3, "type": "VEC3"},
    {"bufferView": 1, "componentType": 5126, "count": len(normals)//3, "type": "VEC3"},
    {"bufferView": 2, "componentType": 5126, "count": len(colors)//4, "type": "VEC4"},
    {"bufferView": 3, "componentType": 5126, "count": len(uvs)//2, "type": "VEC2"},
    {"bufferView": 4, "componentType": 5123, "count": len(indices), "type": "SCALAR"}
]

# add bounding box
position_min = [min(positions[i::3]) for i in range(3)]
position_max = [max(positions[i::3]) for i in range(3)]
accessors[0]["min"] = position_min
accessors[0]["max"] = position_max


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
