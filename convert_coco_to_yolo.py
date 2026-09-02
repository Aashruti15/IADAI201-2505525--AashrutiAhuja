"""
convert_coco_to_yolo.py
------------------------
Your dataset is in COCO format (one big _annotations.coco.json file
per folder). This script converts it into YOLO format (one .txt label
file per image) and creates the data.yaml file that train.py needs.

You only need to run this ONCE.

HOW TO USE:
1. Put this file inside your "pklot-dataset" folder, next to
   test/, train/, valid/ (same level as those folders).
2. Run:  python convert_coco_to_yolo.py
3. It will create a new folder called "yolo_dataset" with the
   correct structure and a data.yaml file, ready for train.py.
"""

import json
import os
import shutil
from pathlib import Path

SOURCE_DIR = Path(".")          # folder containing test/ train/ valid/
OUTPUT_DIR = Path("yolo_dataset")  # where the converted dataset will go
SPLITS = ["train", "valid", "test"]


def convert_split(split_name):
    src_folder = SOURCE_DIR / split_name
    json_path = src_folder / "_annotations.coco.json"

    if not json_path.exists():
        print(f"WARNING: No _annotations.coco.json found in {src_folder}. Skipping.")
        return None

    with open(json_path, "r") as f:
        coco = json.load(f)

    # Map category id -> index (YOLO needs 0-based contiguous class indices)
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_index = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    # Map image id -> image info
    images_by_id = {img["id"]: img for img in coco["images"]}

    # Group annotations by image id
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Output folders
    img_out_dir = OUTPUT_DIR / split_name / "images"
    lbl_out_dir = OUTPUT_DIR / split_name / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    for image_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        src_img_path = src_folder / file_name
        if not src_img_path.exists():
            continue

        # Copy image
        dst_img_path = img_out_dir / file_name
        shutil.copy(src_img_path, dst_img_path)

        # Write YOLO label file (same name, .txt extension)
        label_lines = []
        for ann in anns_by_image.get(image_id, []):
            cat_index = cat_id_to_index[ann["category_id"]]
            x, y, w, h = ann["bbox"]  # COCO format: top-left x,y,width,height

            # Convert to YOLO format: center_x, center_y, width, height (normalized 0-1)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            label_lines.append(f"{cat_index} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
        with open(label_path, "w") as lf:
            lf.write("\n".join(label_lines))

        converted_count += 1

    print(f"{split_name}: converted {converted_count} images")
    return class_names


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_class_names = None

    for split in SPLITS:
        names = convert_split(split)
        if names:
            all_class_names = names

    if all_class_names is None:
        print("ERROR: Could not find any _annotations.coco.json files. "
              "Make sure this script is in the same folder as test/, train/, valid/.")
        return

    # Write data.yaml
    data_yaml_content = f"""train: {OUTPUT_DIR / 'train' / 'images'}
val: {OUTPUT_DIR / 'valid' / 'images'}
test: {OUTPUT_DIR / 'test' / 'images'}

nc: {len(all_class_names)}
names: {all_class_names}
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    print("\nDONE.")
    print(f"Classes found: {all_class_names}")
    print(f"Converted dataset saved to: {OUTPUT_DIR}/")
    print(f"data.yaml created at: {OUTPUT_DIR / 'data.yaml'}")
    print("\nNext: copy train.py so it points to yolo_dataset/data.yaml, then run it.")


if __name__ == "__main__":
    main()