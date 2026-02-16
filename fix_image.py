from PIL import Image

'''
Get-ChildItem -Recurse -Filter *.png | ForEach-Object {
    magick $_.FullName -strip -quality 100 $_.FullName
}
'''

for name in ["native_warrior_rubber", "plants", "treasure", "viking_buildings_hi", "wood_fragments"]:
    try:
        img = Image.open(name + ".png")
        img.save(name + ".png", optimize=True)
    except Exception as e:
        print("Failed:", e)
