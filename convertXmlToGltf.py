import json
import struct
import base64
import math
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from pathlib import Path

bone_to_joint = None # needed by Mesh and Animation. boneName => jointIndex

# TODO: Organize script
# 1. Parse raw data
# 2. Transform data (normalize, rotate, clean, toposort, ...)
# 3. Write to gltf

def warn(msg):
    print(f"\033[38;5;208mWARNING: {msg}\033[0m")

def clean(x, eps=1e-5):
    return 0 if abs(x) < eps else 1 if abs(1-x) < eps else round(x, 6)

def clean_TRS(t, r, s, eps=1e-5):
    t = [clean(x) for x in t]
    r = [clean(x) for x in r]
    s = [clean(x) for x in s]
    return t, r, s

def parseMesh(filename, isSkinned = False):
    tree = ET.parse(filename)
    root = tree.getroot()

    positions = []
    normals = []
    colors = []
    uvs = []
    indices = []

    joints = []
    weights = []

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

    def cleanJointsAndWeights(js, ws):
        # remove 0-weight joints. sum duplicate joints. sort decreasing. add padding
        jw = {}
        for j, w in zip(js, ws):
            if w > 0: jw[j] = jw.get(j,0)+w
        jw = sorted(jw.items(), key=lambda x:-x[1])[:4]
        js_out = [j for j,w in jw] + [0]*(4-len(jw))
        ws_out = [w for j,w in jw] + [0]*(4-len(jw))
        s = sum(ws_out)
        assert(s > 0)
        if s: ws_out = [w/s for w in ws_out]
        else: js_out, ws_out = [js[0],0,0,0],[1,0,0,0]
        return js_out, ws_out

    # Go through polygons
    for polygon in root.find("polygons").findall("polygon"):
        poly_indices = []
        for vertex in polygon.findall("vertex"):
            # Collect vertex attributes
            x, y, z = float(vertex.attrib["x"]), float(vertex.attrib["y"]), float(vertex.attrib["z"])
            nx, ny, nz = float(vertex.attrib["nx"]), float(vertex.attrib["ny"]), float(vertex.attrib["nz"])
            r, g, b, a = float(vertex.attrib["r"]), float(vertex.attrib["g"]), float(vertex.attrib["b"]), float(vertex.attrib["a"])
            u, v = float(vertex.attrib["u"]), float(vertex.attrib["v"])
            
            js = []
            ws = []
            if isSkinned:
                for skin in vertex.findall("skin"):
                    boneName = skin.attrib["bone"]
                    weight = float(skin.attrib["weight"])
                    assert(weight >= 0 and weight <= 1)
                    assert(boneName in bone_to_joint)
                    js.append(bone_to_joint[boneName])
                    ws.append(weight)

                js, ws = cleanJointsAndWeights(js, ws)

            # Transform data
            #x, y, z = rotate(x, y, z) # rotate X -90 deg to align up
            #nx, ny, nz = rotate(nx, ny, nz)
            nx, ny, nz = normalize(nx, ny, nz)
            v = 1.0 - v # flip uv


            key = (x, y, z, nx, ny, nz, r, g, b, a, u, v)
            if isSkinned:
                key = key + (js[0], js[1], js[2], js[3], ws[0], ws[1], ws[2], ws[3])
            if key not in vertex_map:
                vertex_map[key] = current_index
                current_index += 1
                positions.append([x, y, z])
                normals.append([nx, ny, nz])
                colors.append([r, g, b, a])
                uvs.append([u, v])
                joints.append(js)
                weights.append(ws)
            
            poly_indices.append(vertex_map[key])
        
        # Assuming all polygons are triangles
        if len(poly_indices) == 3:
            indices.append(poly_indices)
        elif len(poly_indices) > 3:
            warn('not a triangle - triangulating')
            # Triangulate simple convex polygon (fan method)
            for i in range(1, len(poly_indices)-1):
                indices.append([poly_indices[0], poly_indices[i], poly_indices[i+1]])

    return positions, normals, colors, uvs, indices, joints, weights

def topologicalSort(names, parents, matrices):
    # Topological sort
    topological_index = []

    added = set()
    def topoSort(i):
        if i in added:
            return
        p = parents[i]
        if p != -1:
            topoSort(p)
        topological_index.append(i)
        added.add(i)

    for i in range(len(names)):
        topoSort(i)

    old_to_new = {old: new for new, old in enumerate(topological_index)}

    names_ord         = [names[i]    for i in topological_index]
    #parents_ord       = [parents[i]  for i in topological_index]
    matrices_ord      = [matrices[i] for i in topological_index]
    parents_ord = [
        -1 if parents[i] == -1 else old_to_new[parents[i]]
        for i in topological_index
    ]

    return names_ord, parents_ord, matrices_ord

def parseSkeleton(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    names = []
    parentNames = [] # string
    matrices = []

    for bone in root.find("bones").findall("bone"):
        names.append(bone.attrib["name"])
        parentNames.append(bone.attrib["parent"])

    roots = list(set(parentNames) - set(names))
    if len(roots) != 1:
        warn(f"expected 1 root bone, found {len(roots)}: {roots}")

    for t in root.find("init_pose").findall("transform"):
        name = t.attrib["name"]
        assert(name == names[len(matrices)])

        matrix = [ # ATTENTION: we're reading them transposed since that's how they are in the data
            float(t.attrib["m00"]), float(t.attrib["m10"]), float(t.attrib["m20"]), float(t.attrib["m30"]),
            float(t.attrib["m01"]), float(t.attrib["m11"]), float(t.attrib["m21"]), float(t.attrib["m31"]),
            float(t.attrib["m02"]), float(t.attrib["m12"]), float(t.attrib["m22"]), float(t.attrib["m32"]),
            float(t.attrib["m03"]), float(t.attrib["m13"]), float(t.attrib["m23"]), float(t.attrib["m33"]),
        ]
        matrices.append(matrix)

    name_to_index = {name: i for i, name in enumerate(names)}
    parents = [name_to_index.get(p, -1) for p in parentNames] # indices

    names, parents, matrices = topologicalSort(names, parents, matrices)
    return names, parents, matrices, roots[0]

def parseAnimation(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    frames = []

    for frame in root.findall("frame"):
        index = int(frame.attrib["index"])
        assert(index == len(frames))

        frames.append([None] * len(bone_to_joint))
        for t in frame.findall("transform"):
            name = t.attrib["name"]
            #assert(name == names[len(matrices)])
            matrix = [ # ATTENTION: we're reading them transposed since that's how they are in the data
                float(t.attrib["m00"]), float(t.attrib["m10"]), float(t.attrib["m20"]), float(t.attrib["m30"]),
                float(t.attrib["m01"]), float(t.attrib["m11"]), float(t.attrib["m21"]), float(t.attrib["m31"]),
                float(t.attrib["m02"]), float(t.attrib["m12"]), float(t.attrib["m22"]), float(t.attrib["m32"]),
                float(t.attrib["m03"]), float(t.attrib["m13"]), float(t.attrib["m23"]), float(t.attrib["m33"]),
            ]
            frames[index][bone_to_joint[name]] = matrix
    return frames

def inverseMult(A, B):
    matA = np.array(A, dtype=np.float32).reshape(4,4)
    matB = np.array(B, dtype=np.float32).reshape(4,4)
    matA_inv = np.linalg.inv(matA)
    result = matA_inv @ matB
    result_flat = result.reshape(-1).tolist()
    return result_flat

# TODO: ensure matrix is orthonormal
def invT(A):
    mat = np.array(A, dtype=np.float32).reshape(4,4).T # ATTENTION: transpose needed
    inv = np.linalg.inv(mat)
    inv[np.abs(inv) < 1e-8] = 0.0 # remove noise
    result_flat = inv.reshape(-1).tolist()
    return result_flat

def decompose_matrix(mat4):
    m = np.array(mat4, dtype=np.float32).reshape(4,4)
    t = m[:3, 3].tolist()

    # Extract 3x3 linear part
    M = m[:3, :3].astype(np.float64)

    # Use SVD to robustly separate rotation and scale (polar decomposition)
    U, Svals, Vt = np.linalg.svd(M)
    R_mat = (U @ Vt)
    # Ensure a proper rotation (determinant 1)
    if np.linalg.det(R_mat) < 0:
        U[:, -1] *= -1
        R_mat = U @ Vt

    # Svals are the singular values ~ scale along principal axes
    s = [float(Svals[0]), float(Svals[1]), float(Svals[2])]

    # Convert rotation matrix to quaternion
    r = R.from_matrix(R_mat).as_quat().tolist()

    # convert to python arrays
    t = [float(x) for x in t]
    r = [float(x) for x in r]
    s = [float(x) for x in s]
    return t, r, s


def convertXmlToGltf(main_name, mesh_files, texture_files=None, skeleton_file=None, animation_files=None):
    raw_buffer = b""
    byte_offset = 0

    bufferViews = []
    accessors = []
    meshes = []
    nodes = []
    skins = []
    animations = []
    scene = []


    # root
    nodes.append({
        "name": "Root", # Scene_root
        "rotation": [-0.70710678, 0.0, 0.0, 0.70710678],
        "children": [],
    })
    scene.append(0)

    def addAccessor(values, type, semantic=None, extra=None):
        nonlocal raw_buffer
        nonlocal byte_offset

        if isinstance(values[0], list): # list of vectors        
            values = [f for vec in values for f in vec] # flatten list of vectors to list of floats/ints

        isIndices = type == "SCALARu" # Hacky
        isFloat = type[-1] == "f"
        type = type[:-1]

        N = len(values)
        form = f"<{N}f" if isFloat else f"<{N}H" # <100f = little-endian, 100 floats
        data = struct.pack(form, *values) # bytes array

        # Constants
        ARRAY_BUFFER = 34962
        ELEMENT_ARRAY_BUFFER = 34963
        USHORT = 5123
        FLOAT = 5126

        TYPECOUNTS = {
            "SCALAR": 1,
            "VEC2": 2,
            "VEC3": 3,
            "VEC4": 4,
            "MAT2": 4,
            "MAT3": 9,
            "MAT4": 16,
        }

        alignment = 4 if isFloat else 2
        aligned_offset = (byte_offset + alignment - 1) // alignment * alignment

        #pad with 0s if needed
        padding = aligned_offset - byte_offset
        if padding > 0:
            raw_buffer += b"\x00" * padding
            byte_offset += padding

        raw_buffer += data

        bufferView = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(data),
        }
        if semantic == None:
            bufferView["target"] = ARRAY_BUFFER if not isIndices else ELEMENT_ARRAY_BUFFER
        byte_offset += len(data)

        accessor = {
            "bufferView": len(bufferViews),
            "componentType": FLOAT if isFloat else USHORT, 
            "count": len(values) // TYPECOUNTS[type],
            "type": type,
        }
        if extra:
            accessor.update(extra)

        accessorIndex = len(accessors)
        bufferViews.append(bufferView)
        accessors.append(accessor)
        return accessorIndex

    if skeleton_file:
        names, parents, matrices, rootName = parseSkeleton(skeleton_file)

        '''
        rootJointIdx = len(nodes)
        nodes.append({
            "name": rootName,
            "children": [],
        })
        nodes[0]["children"].append(rootJointIdx)
        '''
        
        global bone_to_joint
        bone_to_joint = {}
        #bone_to_joint = {name: i for i, name in enumerate(names)}
        joints = []
        for i, name in enumerate(names):
            bone_to_joint[name] = i
            joints.append(len(nodes)) # joint to node index
            
            matLocal = inverseMult(matrices[parents[i]], matrices[i]) if parents[i] != -1 else matrices[i]
            nodes.append({
                "name": name,
                "matrix": matLocal,
            })
            if parents[i] != -1:
                parNodeIdx = joints[parents[i]]
                parNode = nodes[parNodeIdx]
                if "children" not in parNode:
                    parNode["children"] = []
                parNode["children"].append(joints[i])
            else:
                nodes[0]["children"].append(joints[i]) # root node is parent of root bone

        inverseBindMatrices = [invT(mat) for mat in matrices]

        skins.append({
            "name": rootName,
            "joints": joints, # indices of nodes that act as bones
            "inverseBindMatrices": addAccessor(inverseBindMatrices, "MAT4f", "IBM"), # accessor of 4x4 matrix
            "skeleton": 0, #joints[0], # node of the hierarchy root
        })
        
    anim_accessors = 0
    for animation_index, filename in enumerate(animation_files):
        frames = parseAnimation(filename)

        time_step = 1/30  # 30fps
        times = [i*time_step for i in range(len(frames))]

        time_accessor = addAccessor(times, "SCALARf", "Time", {"min": [min(times)], "max": [max(times)]})
        animName = Path(filename).stem
        animation = {
            "channels": [],
            "samplers": [],
            "name": animName,
        }

        for bone_index in range(len(bone_to_joint)):  # bones
            i = bone_index
            # collect TRS per frame
            translations, rotations, scales = [], [], []
            for matrices in frames:
                mat = inverseMult(matrices[parents[i]], matrices[i]) if parents[i] != -1 else matrices[i]

                t, r, s = decompose_matrix(mat)
                t, r, s = clean_TRS(t, r, s)
                translations.append(t)
                rotations.append(r)
                scales.append(s)
            
            channels = ["translation", "rotation", "scale"]
            values = [translations, rotations, scales]
            types = ["VEC3f", "VEC4f", "VEC3f"]
            identities = [[0,0,0], [0,0,0,1], [1,1,1]]
            
            for channel, value, type, identity in zip(channels, values, types, identities):
                if all(v == identity for v in value):
                    continue
                accessor_idx = addAccessor(value, type, "Anim")
                sampler_idx = len(animation["samplers"])
                animation["samplers"].append({"input": time_accessor, "output": accessor_idx, "interpolation": "LINEAR"})
                animation["channels"].append({"sampler": sampler_idx, "target": {"node": joints[bone_index], "path": channel}})
                anim_accessors += 1
            
        animations.append(animation)

    print("anim accessors: ", anim_accessors)
    
    for mesh_index, filename in enumerate(mesh_files):
        isSkinned = skeleton_file is not None
        positions, normals, colors, uvs, indices, joints, weights = parseMesh(filename, isSkinned)

        # Bounding box
        position_min = [min(p[i] for p in positions) for i in range(3)]
        position_max = [max(p[i] for p in positions) for i in range(3)]

        attributes = {
            "POSITION":   addAccessor(positions, "VEC3f", None, {"min": position_min, "max": position_max}),
            "NORMAL":     addAccessor(normals, "VEC3f"),
            "COLOR_0":    addAccessor(colors, "VEC4f"),
            "TEXCOORD_0": addAccessor(uvs, "VEC2f"),
        }
        if isSkinned:
            attributes.update({
                "JOINTS_0":   addAccessor(joints, "VEC4u"),
                "WEIGHTS_0":  addAccessor(weights, "VEC4f"),
            })

        meshName = Path(filename).stem
        meshes.append({
            "name": f"Mesh_{meshName}",
            "primitives": [
                {
                    "attributes": attributes,
                    "indices": addAccessor(indices, "SCALARu"),
                }
            ]
        })

        node = {
            "name": f"{meshName}",
            "mesh": mesh_index,
        }
        nodeIdx = len(nodes)
        if isSkinned:
            node["skin"] = 0
            scene.append(nodeIdx)
        else:
            nodes[0]["children"].append(nodeIdx)
        
        nodes.append(node)
            

    #switch nodes
    for i, node in enumerate(nodes):
        if "matrix" in node:
            t, r, s = decompose_matrix(node["matrix"])
            t, r, s = clean_TRS(t, r, s)
            if t != [0, 0, 0]:
                node["translation"] = t
            if r != [0, 0, 0, 1]:
                node["rotation"] = r
            if s != [1, 1, 1]:
                node["scale"] = s
            del node["matrix"]

    # ---- end iteration

    # Build minimal glTF
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{
            "uri": "data:application/octet-stream;base64," + base64.b64encode(raw_buffer).decode("ascii"),
            "byteLength": len(raw_buffer)
        }],
        "bufferViews": bufferViews,
        "accessors": accessors,
        "meshes": meshes,
        "nodes": nodes,
        "skins": skins,
        "animations": animations,
        "scenes": [{"nodes": scene}],
    }
    
    if skins == []:
        del gltf["skins"]
    if animations == []:
        del gltf["animations"]

    #gltf = {k: v for k, v in data.items() if v is not None}

    # Save glTF
    output_file = Path(f"output/{main_name}.gltf")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        json.dump(gltf, f, indent=2)

    #print("Conversion done!")