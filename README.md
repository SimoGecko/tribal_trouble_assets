# Tribal Trouble Assets

This repo provides a script to convert the `xml` based format of Tribal Trouble assets into a `glTF` compliant format for ease of use in modern game engines.
Converted assets are also provided release section.

## Assets

The following assets are available:

#### Vikings
![assets_vikings](images/assets_vikings.png)

#### Natives
![assets_natives](images/assets_natives.png)

#### Misc
![assets_misc](images/assets_misc.png)

- low-poly versions of most assets are included
- buildings have half-built versions also included

## Animations
Characters are skinned, and have a skeleton and multiple animations.

![natives_warrior_run](images/natives_warrior_run.gif)

## Running the script

- download the files locally
- run `py main.py` to create `output/` containing all assets

## Notes
- you might need to adjust the path inside `main.py` to point to your local `tribaltrouble/tt` directory. Download the original files at https://github.com/sunenielsen/tribaltrouble
- some `.xml` skeleton files contain invalid skeletons. To correct them:
	- `peon_skeleton.xml`: replace line 59 `peon bip L Finger0` with `peon bip L Hand`
	- `chicken_skeleton.xml`: replace lines 38-39 `chickenBip R/L Clavicle` with `chickenBip Spine`
- the `geometry.xml` file is used to build and combine the assets. The one under `tt/geometry` works but separates many files out. To use an up-to-date version, use the one in this repository.