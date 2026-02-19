import sys
from pathlib import Path

def check_files(root_folder, reference_file):
    root = Path(root_folder)
    reference_path = Path(reference_file)

    if not root.exists():
        print(f"Root folder not found: {root}")
        return

    if not reference_path.exists():
        print(f"Reference file not found: {reference_path}")
        return

    # Read reference file once
    reference_text = reference_path.read_text(encoding="utf-8")

    missing = []

    for file in root.rglob("*"):
        if file.is_file():
            # Get relative path with forward slashes
            rel_path = file.relative_to(root).as_posix()

            if rel_path not in reference_text:
                missing.append(rel_path)

    print(f"\nChecked files under: {root}")
    print(f"Total files: {len(list(root.rglob('*')))}")
    print(f"Missing references: {len(missing)}\n")

    for m in missing:
        print("Missing:", m)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_files.py <root_folder> <reference_file>")
    else:
        check_files(sys.argv[1], sys.argv[2])
