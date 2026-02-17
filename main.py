from convertXmlToGltf import convertXmlToGltf

geo = 'x:/Github/tribaltrouble/tt/geometry/'
tex = 'x:/Github/tribaltrouble/tt/textures/'

mesh_files = [
	geo + "natives/tower/native_tower_built_hi.xml",
	geo + "natives/tower/native_tower_built_lo.xml",
	geo + "natives/tower/native_tower_build_hi.xml",
	geo + "natives/tower/native_tower_build_lo.xml",
	geo + "natives/tower/native_tower_start_hi.xml",
	geo + "natives/tower/native_tower_start_lo.xml"
]

texture_files = [
    tex + "models/native_tower.png",
    tex + "teamdecals/native_tower_team.png", # NOTES: they have prefix path (should be implicit)
]

mesh_files = [
    geo + "natives/warrior/native_warrior_mesh.xml",
    #geo + "natives/warrior/native_warrior_low_poly_mesh.xml",
]

texture_files = [
    tex + "models/native_warrior_rock.png",
    tex + "teamdecals/native_warrior_rock_team.png",
]

skeleton_file = geo + "natives/warrior/native_warrior_skeleton.xml"
animation_files = [
    #geo + "natives/warrior/native_warrior_idle.xml",
    geo + "natives/warrior/native_warrior_run.xml",
    #geo + "natives/warrior/native_warrior_attack.xml",
    #geo + "natives/warrior/native_warrior_die.xml",
]

'''
<sprite name="warrior">
    <skeleton>natives/warrior/native_warrior_skeleton.xml</skeleton>
    <model r="90" g="60" b="30">
        natives/warrior/native_warrior_mesh.xml
        <texture name="native_warrior_rock" team="native_warrior_rock_team"/>
        <texture name="native_warrior_iron" team="native_warrior_iron_team"/>
        <texture name="native_warrior_rubber" team="native_warrior_rubber_team"/>
    </model>
    <model r="90" g="60" b="30">
        natives/warrior/native_warrior_low_poly_mesh.xml
        <texture name="native_warrior_rock" team="native_warrior_rock_team"/>
        <texture name="native_warrior_iron" team="native_warrior_iron_team"/>
        <texture name="native_warrior_rubber" team="native_warrior_rubber_team"/>
    </model>
    <animation wpc="1" type="loop">natives/warrior/native_warrior_idle.xml</animation>
    <animation wpc="3.2" type="loop">natives/warrior/native_warrior_run.xml</animation>
    <animation wpc="1" type="plain">natives/warrior/native_warrior_attack.xml</animation>
    <animation wpc="1" type="plain">natives/warrior/native_warrior_die.xml</animation>
</sprite>
'''


'''
<sprite name="tower">
    <model r="90" g="60" b="30">
        natives/tower/native_tower_built_hi.xml
        <texture name="native_tower" team="native_tower_team"/>
    </model>
    <model r="90" g="60" b="30">
        natives/tower/native_tower_built_lo.xml
        <texture name="native_tower" team="native_tower_team"/>
    </model>
</sprite>
<sprite name="tower_halfbuilt">
    <model r="90" g="60" b="30">
        natives/tower/native_tower_build_hi.xml
        <texture name="native_tower" team="native_tower_team"/>
    </model>
    <model r="90" g="60" b="30">
        natives/tower/native_tower_build_lo.xml
        <texture name="native_tower" team="native_tower_team"/>
    </model>
</sprite>
<sprite name="tower_start">
    <model r="90" g="60" b="30">
        natives/tower/native_tower_start_hi.xml
        <texture name="native_tower" team="native_tower_team"/>
    </model>
    <model r="90" g="60" b="30">
        natives/tower/native_tower_start_lo.xml
        <texture name="native_tower" team="native_tower_team"/>
    </model>
</sprite>
'''

#convertXmlToGltf("tower", mesh_files, texture_files)
convertXmlToGltf("warrior", mesh_files, texture_files, skeleton_file, animation_files)