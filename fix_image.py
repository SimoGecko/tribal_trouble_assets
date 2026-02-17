from pathlib import Path
from PIL import Image

'''
Get-ChildItem -Recurse -Filter *.png | ForEach-Object {
    magick $_.FullName -strip -quality 100 $_.FullName
}
'''

'''
for name in ["native_warrior_rubber", "plants", "treasure", "viking_buildings_hi", "wood_fragments"]:
    try:
        img = Image.open(name + ".png")
        img.save(name + ".png", optimize=True)
    except Exception as e:
        print("Failed:", e)
'''

start_folder = Path("E:/SABRETOOTH/Users/Simone/Documents/Unity5/Tribal Trouble Assets - Copy/Assets/Textures")  # change to your folder

for img_path in start_folder.rglob("*.png"):
    try:
        img = Image.open(img_path)
        img.save(img_path, optimize=True)
        print(f"Re-encoded: {img_path}")
    except Exception as e:
        print(f"Failed: {img_path} -> {e}")
