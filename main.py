from convertXmlToGltf import convertXmlToGltf
import xml.etree.ElementTree as ET

root = 'x:/Github/tribaltrouble/tt/'

def clean(text):
    return text.strip() if text else ""

def process_geometry(xml_path):
    tree = ET.parse(xml_path)
    rootElem = tree.getroot()

    for group in rootElem.findall("group"):
        group_name = group.attrib["name"]

        for sprite in group.findall("sprite"):
            mesh_files = []
            texture_files = []
            # TODO: texture_teams
            skeleton_file = None
            animation_files = []
            
            sprite_name = sprite.attrib["name"]
            sprite_scale = float(sprite.get("scale", 1.0)) # TODO: use

            # ---- Skeleton ----
            skeleton_elem = sprite.find("skeleton")
            if skeleton_elem is not None:
                skeleton_file = root + "geometry/" + skeleton_elem.text

            # ---- Models ----
            for model in sprite.findall("model"):
                mesh_file = root + "geometry/" + clean(model.text)
                mesh_files.append(mesh_file)

                # Unclear what these are for. Maybe team-color/tint mask?
                r, g, b,  = float(model.attrib["r"]), float(model.attrib["g"]), float(model.attrib["b"])

                # NOTES: this might be dupe since we have multiple models
                for texture in model.findall("texture"):
                    texture_file = root + "texture/models/" + clean(texture.attrib["name"]) + ".png"
                    team_file    = root + "texture/teamdecals/" + clean(texture.get("team", "")) + ".png"
                    texture_files.append(texture_file)

            # ---- Animations ----
            for anim in sprite.findall("animation"):
                wpc = float(anim.get("wpc", 0)) # ?? maybe Walk per cycle? / World units per cycle
                type = anim.get("type", "") # loop or plain
                assert(type == "loop" or type == "plain")
                animation_file = root + "geometry/" + clean(anim.text)
                animation_files.append(animation_file)

            # ---- Call asset creation ----
            asset_name = group_name + "_" + sprite_name
            if asset_name != "natives_warrior":
                continue
            #print(asset_name, mesh_files, texture_files, skeleton_file, animation_files)
            print(f"Processing {asset_name}...")
            convertXmlToGltf(asset_name, mesh_files, texture_files, skeleton_file, animation_files)

process_geometry(root + "geometry/geometry.xml")