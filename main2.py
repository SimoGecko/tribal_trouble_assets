from convertXmlToGltf import convertXmlToGltf

path = 'x:/Github/tribaltrouble/tt/geometry/'

mesh_files = [
	path + "natives/tower/native_tower_build_hi.xml",
	path + "natives/tower/native_tower_build_lo.xml",
	path + "natives/tower/native_tower_built_hi.xml",
	path + "natives/tower/native_tower_built_lo.xml",
	path + "natives/tower/native_tower_start_hi.xml",
	path + "natives/tower/native_tower_start_lo.xml"
]

convertXmlToGltf("tower", mesh_files)