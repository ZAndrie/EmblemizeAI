# organize_car_logo_dataset.py - Organize Car Logo Dataset for Training
import os
import shutil
from pathlib import Path
from PIL import Image

print("=" * 60)
print("Car Logo Dataset Organizer")
print("=" * 60)

# Find the Car_Logo Dataset folder
possible_names = [
    "Car_Logo Dataset",
    "Car_Logo_Dataset",
    "Car Logo Dataset",
    "car_logo_dataset",
    "Car_Logo",
    "car-logo-dataset"
]

source_folder = None
for name in possible_names:
    if os.path.exists(name):
        source_folder = Path(name)
        print(f"\n✓ Found dataset: {source_folder}")
        break

if not source_folder:
    print("\n✗ Car Logo Dataset folder not found!")
    print("\nPlease ensure you have one of these folders:")
    for name in possible_names:
        print(f"  - {name}/")
    print("\nCurrent directory contents:")
    for item in os.listdir("."):
        print(f"  - {item}")
    exit(1)

# Explore the structure
print("\n" + "=" * 60)
print("Analyzing dataset structure...")
print("=" * 60)


def explore_folder(path, level=0):
    """Recursively explore folder structure"""
    items = list(path.iterdir())
    folders = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]

    indent = "  " * level
    print(f"{indent}📁 {path.name}/")

    if folders:
        print(f"{indent}  Subfolders: {len(folders)}")
        if level < 2:  # Only show first 2 levels
            for folder in folders[:5]:  # Show first 5
                explore_folder(folder, level + 1)
            if len(folders) > 5:
                print(f"{indent}  ... and {len(folders) - 5} more")

    if files:
        images = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        if images:
            print(f"{indent}  Images: {len(images)}")


explore_folder(source_folder)

# Find Train and Test folders
print("\n" + "=" * 60)
print("Looking for Train/Test folders...")
print("=" * 60)

train_path = None
test_path = None

# Search in the source folder and subfolders
for item in source_folder.rglob("*"):
    if item.is_dir():
        name_lower = item.name.lower()
        if name_lower in ["train", "training"]:
            train_path = item
            print(f"✓ Found Train: {item}")
        elif name_lower in ["test", "testing", "validation", "val"]:
            test_path = item
            print(f"✓ Found Test: {item}")

if not train_path:
    # Maybe the brands are directly in the source folder
    print("\n⚠ No Train folder found. Checking for brand folders directly...")
    potential_brands = [d for d in source_folder.iterdir() if d.is_dir()]

    if potential_brands:
        print(f"✓ Found {len(potential_brands)} potential brand folders:")
        for brand in potential_brands[:10]:
            img_count = len(list(brand.glob("*.jpg")) + list(brand.glob("*.png")) + list(brand.glob("*.jpeg")))
            print(f"  - {brand.name}: {img_count} images")
        if len(potential_brands) > 10:
            print(f"  ... and {len(potential_brands) - 10} more")

        train_path = source_folder
    else:
        print("✗ Could not find brand folders!")
        exit(1)

# Create organized dataset structure
print("\n" + "=" * 60)
print("Creating organized dataset folder...")
print("=" * 60)

dataset_dir = Path("dataset")

# Ask before overwriting
if dataset_dir.exists():
    print(f"\n⚠ Warning: 'dataset' folder already exists!")
    response = input("Delete and recreate? (yes/no): ").strip().lower()
    if response == 'yes':
        shutil.rmtree(dataset_dir)
        print("✓ Deleted old dataset folder")
    else:
        print("Keeping existing dataset. Exiting.")
        exit(0)

dataset_dir.mkdir(exist_ok=True)
print(f"✓ Created: {dataset_dir.absolute()}")

# Get all brand folders
brand_folders = [f for f in train_path.iterdir() if f.is_dir()]

if not brand_folders:
    print("✗ No brand folders found in Train directory!")
    exit(1)

print(f"\n✓ Found {len(brand_folders)} car brands")

# Copy images to organized structure
print("\n" + "=" * 60)
print("Copying and organizing images...")
print("=" * 60)

total_images = 0
brand_stats = {}

for brand_folder in sorted(brand_folders):
    brand_name = brand_folder.name
    dest_folder = dataset_dir / brand_name
    dest_folder.mkdir(exist_ok=True)

    # Collect all images from train
    train_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
        train_images.extend(list(brand_folder.glob(ext)))

    # Also collect from test if exists
    if test_path:
        test_brand_folder = test_path / brand_name
        if test_brand_folder.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']:
                train_images.extend(list(test_brand_folder.glob(ext)))

    # Copy images with validation
    copied = 0
    for i, img_path in enumerate(train_images):
        try:
            # Verify it's a valid image
            img = Image.open(img_path)
            img.verify()

            # Copy with new name
            dest_path = dest_folder / f"{brand_name}_{i:04d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dest_path)
            copied += 1

        except Exception as e:
            print(f"  ⚠ Skipped corrupted image: {img_path.name}")
            continue

    brand_stats[brand_name] = copied
    total_images += copied
    print(f"✓ {brand_name}: {copied} images")

# Create classes.txt
classes_file = Path("classes.txt")
with open(classes_file, 'w') as f:
    for brand_name in sorted(brand_stats.keys()):
        f.write(f"{brand_name}\n")

print(f"\n✓ Created {classes_file}")

# Summary
print("\n" + "=" * 60)
print("Dataset Organization Complete! 🎉")
print("=" * 60)

print(f"\nDataset Summary:")
print(f"  Location: {dataset_dir.absolute()}")
print(f"  Total Brands: {len(brand_stats)}")
print(f"  Total Images: {total_images}")
print(f"  Average per brand: {total_images // len(brand_stats) if brand_stats else 0}")

print(f"\nBrand Distribution:")
sorted_brands = sorted(brand_stats.items(), key=lambda x: x[1], reverse=True)
for brand, count in sorted_brands:
    bar = "█" * (count // 10) if count >= 10 else "▌"
    print(f"  {brand:20s} {count:4d} images {bar}")

# Check for imbalanced classes
min_images = min(brand_stats.values())
max_images = max(brand_stats.values())
if max_images > min_images * 3:
    print(f"\n⚠ Warning: Dataset is imbalanced!")
    print(f"  Smallest class: {min_images} images")
    print(f"  Largest class: {max_images} images")
    print(f"  Consider data augmentation during training")

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("1. Review the dataset folder structure")
print("2. Train the model:")
print("   python train.py")
print("3. Test the web app:")
print("   python app.py")
print("4. (Optional) Convert for mobile:")
print("   python convert_tflite.py")
print("   python main.py")
print("=" * 60)

print(f"\n💡 Tip: The training script will automatically split data into")
print(f"   85% training and 15% validation sets")