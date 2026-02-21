from pathlib import Path
from PIL import Image

'''
Get-ChildItem -Recurse -Filter *.png | ForEach-Object {
    magick $_.FullName -strip -quality 100 $_.FullName
}
'''

root = "x:/Github/tribaltrouble/tt/textures"

for img_path in Path(root).rglob("*.png"):
    try:
        img = Image.open(img_path)
        img.save(img_path, optimize=True)
        print(f"Re-encoded: {img_path}")
    except Exception as e:
        print(f"Failed: {img_path} -> {e}")
