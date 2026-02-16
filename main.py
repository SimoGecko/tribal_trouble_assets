import xml.etree.ElementTree as ET
import json
import struct
import base64
import math

buffers_data = []
bufferViews = []
accessors = []
meshes = []
nodes = []
bufferViews_index = 0
accessors_index = 0
byte_offset = 0

xml_files = ["chicken_mesh.xml", "treasure_2.xml"]

for mesh_index, filename in enumerate(xml_files):
    # Load XML
    tree = ET.parse(filename)
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

    def normalize(x, y, z):
        length = math.sqrt(x**2 + y**2 + z**2)
        if length > 0:
            x /= length
            y /= length
            z /= length
        return x, y, z

    # Go through polygons
    for polygon in root.find("polygons").findall("polygon"):
        poly_indices = []
        for vertex in polygon.findall("vertex"):
            # Collect vertex attributes
            x, y, z = float(vertex.attrib["x"]), float(vertex.attrib["y"]), float(vertex.attrib["z"])
            nx, ny, nz = float(vertex.attrib["nx"]), float(vertex.attrib["ny"]), float(vertex.attrib["nz"])
            r, g, b, a = float(vertex.attrib["r"]), float(vertex.attrib["g"]), float(vertex.attrib["b"]), float(vertex.attrib["a"])
            u, v = float(vertex.attrib["u"]), float(vertex.attrib["v"])
            
            # Transform data
            x, y, z = rotate(x, y, z)
            nx, ny, nz = rotate(nx, ny, nz)
            nx, ny, nz = normalize(nx, ny, nz)
            v = 1.0 - v # flip uv

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

    # Helper to convert float lists to base64
    def float_to_b64(floats):
        b = struct.pack("<" + "f"*len(floats), *floats)
        return base64.b64encode(b).decode('ascii')

    def uint16_to_b64(uints):
        b = struct.pack("<" + "H"*len(uints), *uints)
        return base64.b64encode(b).decode('ascii')

    # Create buffer data
    position_b64 = float_to_b64(positions)
    normal_b64 = float_to_b64(normals)
    color_b64 = float_to_b64(colors)
    uv_b64 = float_to_b64(uvs)
    index_b64 = uint16_to_b64(indices)

    # Each bufferView references data inside a buffer (all inline here)
    buffers_dataL = [
        position_b64, normal_b64, color_b64, uv_b64, index_b64
    ]

    buffers_data.extend(buffers_dataL)

    # Constants
    ARRAY_BUFFER = 34962
    ELEMENT_ARRAY_BUFFER = 34963
    USHORT = 5123
    FLOAT = 5126

    # Precompute decoded lengths
    position_bytes = len(base64.b64decode(position_b64))
    normal_bytes   = len(base64.b64decode(normal_b64))
    color_bytes    = len(base64.b64decode(color_b64))
    uv_bytes       = len(base64.b64decode(uv_b64))
    index_bytes    = len(base64.b64decode(index_b64))

    def offset(value):
        nonlocal byte_offset
        byte_offset += value
        return byte_offset-value

    bufferViews.extend([
        {"buffer": 0, "byteOffset": offset(position_bytes), "byteLength": position_bytes, "target": ARRAY_BUFFER},
        {"buffer": 0, "byteOffset": offset(normal_bytes),   "byteLength": normal_bytes,   "target": ARRAY_BUFFER},
        {"buffer": 0, "byteOffset": offset(color_bytes),    "byteLength": color_bytes,    "target": ARRAY_BUFFER},  # COLOR_0
        {"buffer": 0, "byteOffset": offset(uv_bytes),       "byteLength": uv_bytes,       "target": ARRAY_BUFFER},  # TEXCOORD_0
        {"buffer": 0, "byteOffset": offset(index_bytes),    "byteLength": index_bytes,    "target": ELEMENT_ARRAY_BUFFER},
    ])



    '''
    for i, data in enumerate(buffers_dataL):
        bufferViews.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(base64.b64decode(data))
        })
        byte_offset += len(base64.b64decode(data))

    bufferViews[bufferViews_index+0]["target"] = ARRAY_BUFFER
    bufferViews[bufferViews_index+1]["target"] = ARRAY_BUFFER
    bufferViews[bufferViews_index+2]["target"] = ARRAY_BUFFER
    bufferViews[bufferViews_index+3]["target"] = ARRAY_BUFFER
    bufferViews[bufferViews_index+4]["target"] = ELEMENT_ARRAY_BUFFER
    '''
    '''
    bufferViews = [
        {"buffer":0, "byteOffset":0, "byteLength":len(base64.b64decode(position_b64)), "target": ARRAY_BUFFER},
        {"buffer":0, "byteOffset":len(base64.b64decode(position_b64)), "byteLength":len(base64.b64decode(normal_b64)), "target": ARRAY_BUFFER},
        {"buffer":0, "byteOffset":..., "byteLength":..., "target": ARRAY_BUFFER},  # for COLOR_0
        {"buffer":0, "byteOffset":..., "byteLength":..., "target": ARRAY_BUFFER},  # for TEXCOORD_0
        {"buffer":0, "byteOffset":..., "byteLength":len(base64.b64decode(index_b64)), "target": ELEMENT_ARRAY_BUFFER},
    ]
    '''

    # add bounding box
    position_min = [min(positions[i::3]) for i in range(3)]
    position_max = [max(positions[i::3]) for i in range(3)]

    # Accessors (simple)
    accessors.extend([
        {"bufferView": bufferViews_index+0, "componentType": FLOAT,  "count": len(positions)//3, "type": "VEC3", "min": position_min, "max": position_max},
        {"bufferView": bufferViews_index+1, "componentType": FLOAT,  "count": len(normals)//3,   "type": "VEC3"},
        {"bufferView": bufferViews_index+2, "componentType": FLOAT,  "count": len(colors)//4,    "type": "VEC4"},
        {"bufferView": bufferViews_index+3, "componentType": FLOAT,  "count": len(uvs)//2,       "type": "VEC2"},
        {"bufferView": bufferViews_index+4, "componentType": USHORT, "count": len(indices),      "type": "SCALAR"}
    ])

    meshes.append({
        "name": f"Mesh_{mesh_index}",
        "primitives": [
            {
                "attributes": {
                    "POSITION":   accessors_index+0,
                    "NORMAL":     accessors_index+1,
                    "COLOR_0":    accessors_index+2,
                    "TEXCOORD_0": accessors_index+3,
                },
                "indices": accessors_index+4,
            }
        ]
    })

    nodes.append({
        "name": f"Node_{mesh_index}",
        "mesh": mesh_index,
        #"translation": [0,0,0]  # optional, can move each mesh
    })

    bufferViews_index += 5
    accessors_index += 5


#### --------- done

# Build minimal glTF
gltf = {
    "asset": {"version": "2.0"},
    "buffers": [],
    "bufferViews": bufferViews,
    "accessors": accessors,
    "meshes": meshes,
    "nodes": nodes,
    #"scenes": {"nodes": list(range(len(nodes)))}, #[{"nodes": [0]}],
    "scenes": [{"nodes": list(range(len(nodes)))}],
    #"scene": 0,
    #"scenes": [{"nodes": [0]}],
}


gltf["buffers"].append({
    "uri": "data:application/octet-stream;base64," + "".join(buffers_data),
    "byteLength": byte_offset
})

# Save glTF
with open("mesh.gltf", "w") as f:
    json.dump(gltf, f, indent=2)

print("Conversion done!")
