import xml.etree.ElementTree as ET
import json
import struct
import base64
import math
from pathlib import Path

def parseMesh(filename):
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

    return positions, normals, colors, uvs, indices

def parseSkeleton(filename):
    pass

def parseAnimation(filename):
    pass

def convertXmlToGltf(name, mesh_files, texture_files=None, skeleton_file=None, animation_files=None):
    raw_buffer = b""

    bufferViews = []
    accessors = []
    meshes = []
    nodes = []

    bufferViews_index = 0
    accessors_index = 0
    byte_offset = 0

    for mesh_index, filename in enumerate(mesh_files):
        positions, normals, colors, uvs, indices = parseMesh(filename)

        # Helper to convert float/uint lists to bytes
        def float_to_bytes(floats):
            form = f"<{len(floats)}f" # <100f = little-endian, 100 floats
            b = struct.pack(form, *floats)
            return b

        def uint16_to_bytes(uints):
            form = f"<{len(uints)}H"
            b = struct.pack(form, *uints)
            return b

        # Create buffer data
        position_b = float_to_bytes(positions)
        normal_b   = float_to_bytes(normals)
        color_b    = float_to_bytes(colors)
        uv_b       = float_to_bytes(uvs)
        index_b    = uint16_to_bytes(indices)

        # Each bufferView references data inside a buffer (all inline here)
        buffers_data = [
            position_b, normal_b, color_b, uv_b, index_b
        ]

        # Constants
        ARRAY_BUFFER = 34962
        ELEMENT_ARRAY_BUFFER = 34963
        USHORT = 5123
        FLOAT = 5126

        def align(n, alignment):
            return (n + alignment - 1) // alignment * alignment

        # TODO: expand, just like accessors below
        for i, data in enumerate(buffers_data):
            isFloat = i!=4 # else ushort
            aligned_offset = align(byte_offset, 4 if isFloat else 2)

            #pad with 0s if needed
            padding = aligned_offset - byte_offset
            if padding > 0:
                raw_buffer += b"\x00" * padding
                byte_offset += padding

            raw_buffer += data

            bufferViews.append({
                "buffer": 0,
                "byteOffset": byte_offset,
                "byteLength": len(data),
                "target": ARRAY_BUFFER if isFloat else ELEMENT_ARRAY_BUFFER,
            })
            byte_offset += len(data)

        # Bounding box
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

        meshName = Path(filename).stem
        meshes.append({
            "name": f"Mesh_{meshName}",
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
            "name": f"{meshName}",
            "mesh": mesh_index,
            #"translation": [0,0,0]  # optional, can move each mesh
        })

        bufferViews_index += 5
        accessors_index += 5


    # ---- end iteration

    # Build minimal glTF
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [],
        "bufferViews": bufferViews,
        "accessors": accessors,
        "meshes": meshes,
        "nodes": nodes,
        "scenes": [{"nodes": list(range(len(nodes)))}],
    }

    gltf["buffers"].append({
        "uri": "data:application/octet-stream;base64," + base64.b64encode(raw_buffer).decode("ascii"),
        "byteLength": len(raw_buffer)
    })

    # Save glTF
    with open(f"{name}.gltf", "w") as f:
        json.dump(gltf, f, indent=2)

    print("Conversion done!")