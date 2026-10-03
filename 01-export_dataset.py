import json
import shutil
import sys
from itertools import combinations
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


# =========================================================
# SETTINGS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = SCRIPT_DIR / "images"
OUTPUT_DIR = SCRIPT_DIR / "dataset"

TARGET_VAL_RATIO = 0.20

CLASS_TO_ID = {
    "1baht": 0,
    "2baht": 1,
    "5baht": 2,
    "10baht": 3,
}

CLASSES = [
    "1baht",
    "2baht",
    "5baht",
    "10baht",
]


# =========================================================
# FIND JSON
# =========================================================

def find_json_file(folder: Path) -> Path:
    json_files = sorted(folder.glob("*.json"))

    if not json_files:
        sys.exit(f"ERROR: ไม่พบไฟล์ .json ใน {folder}")

    if len(json_files) > 1:
        print("WARNING: พบ JSON มากกว่า 1 ไฟล์")
        print("จะใช้ไฟล์แรก:")
        for f in json_files:
            print(f"  - {f.name}")

    return json_files[0]


# =========================================================
# GET IMAGE PATH
# =========================================================

def get_image_relative_path(task):
    data = task.get("data", {})
    image_value = data.get("image")

    if not image_value:
        return None

    parsed = urlparse(image_value)
    query = parse_qs(parsed.query)

    if "d" in query:
        rel_path = unquote(query["d"][0])
        rel_path = rel_path.replace("\\", "/")

        if rel_path.lower().startswith("images/"):
            rel_path = rel_path[len("images/"):]

        return Path(rel_path)

    return Path(unquote(parsed.path).lstrip("/"))


# =========================================================
# GET LABELS
# =========================================================

def get_task_labels(task):
    labels_found = []

    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue

        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue

            labels = result.get("value", {}).get(
                "rectanglelabels", []
            )

            for label in labels:
                if label in CLASS_TO_ID:
                    labels_found.append(label)

    return labels_found


# =========================================================
# CONVERT TO YOLO
# =========================================================

def convert_task_to_yolo_lines(task):
    lines = []

    for annotation in task.get("annotations", []):
        if annotation.get("was_cancelled"):
            continue

        for result in annotation.get("result", []):
            if result.get("type") != "rectanglelabels":
                continue

            value = result.get("value", {})
            labels = value.get("rectanglelabels", [])

            if not labels:
                continue

            x = value.get("x")
            y = value.get("y")
            w = value.get("width")
            h = value.get("height")

            if None in (x, y, w, h):
                continue

            x_center = (x + w / 2) / 100.0
            y_center = (y + h / 2) / 100.0
            width = w / 100.0
            height = h / 100.0

            for label in labels:
                if label not in CLASS_TO_ID:
                    continue

                class_id = CLASS_TO_ID[label]

                lines.append(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

    return lines


# =========================================================
# ANALYZE SPLIT
# =========================================================

def analyze_split(
    val_group_names,
    groups,
    total_images,
    total_boxes_per_class,
):
    val_images = 0
    val_single = 0
    val_multi = 0

    val_boxes = {
        cls: 0
        for cls in CLASSES
    }

    for group_name in val_group_names:
        group = groups[group_name]

        val_images += group["image_count"]
        val_single += group["single_object_images"]
        val_multi += group["multi_object_images"]

        for cls in CLASSES:
            val_boxes[cls] += group["box_counts"][cls]

    val_ratio = val_images / total_images

    class_ratios = {}

    for cls in CLASSES:
        total_cls = total_boxes_per_class[cls]

        if total_cls == 0:
            class_ratios[cls] = 0.0
        else:
            class_ratios[cls] = val_boxes[cls] / total_cls

    return {
        "val_images": val_images,
        "val_ratio": val_ratio,
        "val_single": val_single,
        "val_multi": val_multi,
        "val_boxes": val_boxes,
        "class_ratios": class_ratios,
    }


# =========================================================
# SCORE SPLIT
# =========================================================

def score_split(stats):
    score = 0.0

    # อยากให้จำนวนภาพ val ใกล้ 20%
    score += abs(
        stats["val_ratio"] - TARGET_VAL_RATIO
    ) * 3.0

    # อยากให้แต่ละ class ใกล้ 20%
    for cls in CLASSES:
        score += abs(
            stats["class_ratios"][cls]
            - TARGET_VAL_RATIO
        ) * 2.0

    # ต้องมีทั้ง single และ multi
    if stats["val_single"] == 0:
        score += 5.0

    if stats["val_multi"] == 0:
        score += 5.0

    # ไม่อยากให้ val น้อยกว่า 10%
    if stats["val_ratio"] < 0.10:
        score += 5.0

    # ไม่อยากให้ val มากกว่า 35%
    if stats["val_ratio"] > 0.35:
        score += 5.0

    return score


# =========================================================
# FIND BEST VIDEO SPLIT
# =========================================================

def find_best_video_split(groups):
    group_names = list(groups.keys())

    total_images = sum(
        groups[g]["image_count"]
        for g in group_names
    )

    total_boxes_per_class = {
        cls: sum(
            groups[g]["box_counts"][cls]
            for g in group_names
        )
        for cls in CLASSES
    }

    print()
    print("=" * 70)
    print("SEARCHING BEST TRAIN / VAL SPLIT")
    print("=" * 70)

    print(f"Video groups : {len(group_names)}")
    print(f"Total images : {total_images}")

    best_score = float("inf")
    best_val_groups = None
    best_stats = None

    candidates_checked = 0

    for r in range(1, len(group_names)):

        for combo in combinations(group_names, r):
            candidates_checked += 1

            stats = analyze_split(
                combo,
                groups,
                total_images,
                total_boxes_per_class,
            )

            score = score_split(stats)

            if score < best_score:
                best_score = score
                best_val_groups = set(combo)
                best_stats = stats

    if best_val_groups is None:
        sys.exit("ERROR: ไม่สามารถสร้าง validation split ได้")

    train_groups = set(group_names) - best_val_groups

    print()
    print(f"Candidates checked : {candidates_checked}")
    print(f"Best score         : {best_score:.4f}")

    print()
    print("BEST SPLIT:")
    print(
        f"Val images : "
        f"{best_stats['val_images']} "
        f"({best_stats['val_ratio'] * 100:.1f}%)"
    )

    print(
        f"Val single : {best_stats['val_single']}"
    )

    print(
        f"Val multi  : {best_stats['val_multi']}"
    )

    print()

    for cls in CLASSES:
        print(
            f"{cls:<7} "
            f"Val boxes={best_stats['val_boxes'][cls]} "
            f"({best_stats['class_ratios'][cls] * 100:.1f}%)"
        )

    return train_groups, best_val_groups


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("THAI COIN YOLO DATASET EXPORTER")
    print("BEST-EFFORT VIDEO SPLIT")
    print("=" * 70)

    json_path = find_json_file(SCRIPT_DIR)

    print()
    print("JSON:")
    print(json_path)

    if not IMAGES_DIR.exists():
        sys.exit(
            f"ERROR: ไม่พบ images folder: {IMAGES_DIR}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    valid_tasks = []

    missing_images = 0
    no_annotations = 0

    for task in tasks:

        relative_path = get_image_relative_path(task)

        if relative_path is None:
            continue

        source_image = IMAGES_DIR / relative_path

        if not source_image.exists():
            missing_images += 1
            continue

        labels = get_task_labels(task)

        if not labels:
            no_annotations += 1
            continue

        valid_tasks.append({
            "task": task,
            "relative_path": relative_path,
            "source_image": source_image,
            "labels": labels,
        })

    print()
    print(f"Valid images   : {len(valid_tasks)}")
    print(f"Missing images : {missing_images}")
    print(f"No annotation  : {no_annotations}")

    # =====================================================
    # GROUP BY VIDEO
    # =====================================================

    groups = {}

    for item in valid_tasks:

        relative_path = item["relative_path"]
        group_name = relative_path.parent.as_posix()

        if group_name not in groups:
            groups[group_name] = {
                "items": [],
                "image_count": 0,
                "single_object_images": 0,
                "multi_object_images": 0,
                "box_count": 0,
                "box_counts": {
                    cls: 0
                    for cls in CLASSES
                },
            }

        group = groups[group_name]

        group["items"].append(item)
        group["image_count"] += 1

        num_boxes = len(item["labels"])

        group["box_count"] += num_boxes

        if num_boxes == 1:
            group["single_object_images"] += 1
        elif num_boxes > 1:
            group["multi_object_images"] += 1

        for label in item["labels"]:
            group["box_counts"][label] += 1

    # =====================================================
    # SHOW GROUPS
    # =====================================================

    print()
    print("=" * 70)
    print("VIDEO GROUPS")
    print("=" * 70)

    for group_name, info in sorted(groups.items()):

        print(
            f"{group_name:<12} "
            f"img={info['image_count']:3} "
            f"single={info['single_object_images']:3} "
            f"multi={info['multi_object_images']:3} "
            f"boxes={info['box_count']:3} "
            f"1b={info['box_counts']['1baht']:3} "
            f"2b={info['box_counts']['2baht']:3} "
            f"5b={info['box_counts']['5baht']:3} "
            f"10b={info['box_counts']['10baht']:3}"
        )

    # =====================================================
    # SPLIT
    # =====================================================

    train_groups, val_groups = find_best_video_split(
        groups
    )

    print()
    print("=" * 70)
    print("TRAIN VIDEO GROUPS")
    print("=" * 70)

    for group in sorted(train_groups):
        print("TRAIN :", group)

    print()
    print("=" * 70)
    print("VAL VIDEO GROUPS")
    print("=" * 70)

    for group in sorted(val_groups):
        print("VAL   :", group)

    # =====================================================
    # BUILD ITEM LISTS
    # =====================================================

    train_items = []
    val_items = []

    for group_name, info in groups.items():

        if group_name in train_groups:
            train_items.extend(info["items"])

        elif group_name in val_groups:
            val_items.extend(info["items"])

    # =====================================================
    # REMOVE OLD DATASET
    # =====================================================

    if OUTPUT_DIR.exists():
        print()
        print("Removing old dataset...")
        shutil.rmtree(OUTPUT_DIR)

    # =====================================================
    # CREATE DIRS
    # =====================================================

    for split in ("train", "val"):

        (
            OUTPUT_DIR
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_DIR
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # =====================================================
    # EXPORT
    # =====================================================

    def process_split(items, split_name):

        stats = {
            "images": 0,
            "boxes": 0,
            "single": 0,
            "multi": 0,

            "class_images": {
                cls: 0
                for cls in CLASSES
            },

            "class_boxes": {
                cls: 0
                for cls in CLASSES
            },
        }

        for item in items:

            task = item["task"]
            relative_path = item["relative_path"]
            source_image = item["source_image"]
            labels = item["labels"]

            yolo_lines = convert_task_to_yolo_lines(
                task
            )

            if not yolo_lines:
                continue

            filename = "_".join(
                relative_path.parts
            )

            dst_image = (
                OUTPUT_DIR
                / "images"
                / split_name
                / filename
            )

            shutil.copy2(
                source_image,
                dst_image
            )

            dst_label = (
                OUTPUT_DIR
                / "labels"
                / split_name
                / Path(filename).with_suffix(".txt")
            )

            dst_label.write_text(
                "\n".join(yolo_lines) + "\n",
                encoding="utf-8"
            )

            stats["images"] += 1
            stats["boxes"] += len(yolo_lines)

            if len(yolo_lines) == 1:
                stats["single"] += 1
            else:
                stats["multi"] += 1

            unique_classes = set(labels)

            for cls in labels:
                stats["class_boxes"][cls] += 1

            for cls in unique_classes:
                stats["class_images"][cls] += 1

        return stats

    train_stats = process_split(
        train_items,
        "train"
    )

    val_stats = process_split(
        val_items,
        "val"
    )

    # =====================================================
    # WRITE CLASSES
    # =====================================================

    (
        OUTPUT_DIR / "classes.txt"
    ).write_text(
        "\n".join(CLASSES) + "\n",
        encoding="utf-8"
    )

    # =====================================================
    # WRITE YAML
    # =====================================================

    yaml_content = f"""path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: 1baht
  1: 2baht
  2: 5baht
  3: 10baht
"""

    (
        OUTPUT_DIR / "data.yaml"
    ).write_text(
        yaml_content,
        encoding="utf-8"
    )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    total_images = (
        train_stats["images"]
        + val_stats["images"]
    )

    total_boxes = (
        train_stats["boxes"]
        + val_stats["boxes"]
    )

    print()
    print("=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    print()

    print(
        f"Train images : "
        f"{train_stats['images']} "
        f"({train_stats['images'] / total_images * 100:.1f}%)"
    )

    print(
        f"Val images   : "
        f"{val_stats['images']} "
        f"({val_stats['images'] / total_images * 100:.1f}%)"
    )

    print()

    print(
        f"Train boxes  : "
        f"{train_stats['boxes']} "
        f"({train_stats['boxes'] / total_boxes * 100:.1f}%)"
    )

    print(
        f"Val boxes    : "
        f"{val_stats['boxes']} "
        f"({val_stats['boxes'] / total_boxes * 100:.1f}%)"
    )

    print()

    print(
        f"Train single-object : "
        f"{train_stats['single']}"
    )

    print(
        f"Train multi-object  : "
        f"{train_stats['multi']}"
    )

    print(
        f"Val single-object   : "
        f"{val_stats['single']}"
    )

    print(
        f"Val multi-object    : "
        f"{val_stats['multi']}"
    )

    print()
    print("=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    print()

    print(
        f"{'Class':<10}"
        f"{'Train img':>11}"
        f"{'Val img':>10}"
        f"{'Train box':>12}"
        f"{'Val box':>10}"
        f"{'Val %':>9}"
    )

    print("-" * 62)

    for cls in CLASSES:

        train_boxes = train_stats[
            "class_boxes"
        ][cls]

        val_boxes = val_stats[
            "class_boxes"
        ][cls]

        total_cls = train_boxes + val_boxes

        val_pct = (
            val_boxes / total_cls * 100
            if total_cls
            else 0
        )

        print(
            f"{cls:<10}"
            f"{train_stats['class_images'][cls]:>11}"
            f"{val_stats['class_images'][cls]:>10}"
            f"{train_boxes:>12}"
            f"{val_boxes:>10}"
            f"{val_pct:>8.1f}%"
        )

    print()
    print("Class IDs:")

    for cls in CLASSES:
        print(
            f"{CLASS_TO_ID[cls]} = {cls}"
        )

    print()
    print("Dataset:")
    print(OUTPUT_DIR)

    print()
    print("data.yaml:")
    print(OUTPUT_DIR / "data.yaml")

    print()
    print("พร้อม Train ใหม่")


if __name__ == "__main__":
    main()