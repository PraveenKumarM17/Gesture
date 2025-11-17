"""
Quick script to check if training data is balanced
"""

from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / 'data_grayscale' / 'train'

print("\n" + "=" * 80)
print("📊 DATA BALANCE CHECKER")
print("=" * 80)

if not TRAIN_DIR.exists():
    print(f"❌ Directory not found: {TRAIN_DIR}")
    exit(1)

class_counts = {}

for class_folder in sorted(TRAIN_DIR.iterdir()):
    if class_folder.is_dir():
        image_files = (
            list(class_folder.glob('*.jpg')) +
            list(class_folder.glob('*.png')) +
            list(class_folder.glob('*.jpeg')) +
            list(class_folder.glob('*.JPG')) +
            list(class_folder.glob('*.PNG'))
        )
        class_counts[class_folder.name] = len(image_files)

if not class_counts:
    print("❌ No classes found!")
    exit(1)

total = sum(class_counts.values())
max_count = max(class_counts.values())
min_count = min(class_counts.values())
avg_count = total / len(class_counts)

print(f"\n📈 STATISTICS:")
print(f"   Total classes: {len(class_counts)}")
print(f"   Total images: {total}")
print(f"   Average per class: {avg_count:.1f}")
print(f"   Max: {max_count} | Min: {min_count}")
print(f"   Imbalance ratio: {max_count/min_count:.2f}x")

print(f"\n📊 CLASS DISTRIBUTION:")
print("-" * 80)

# Sort by count
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

for class_name, count in sorted_classes:
    bar_length = int((count / max_count) * 40)
    bar = '█' * bar_length
    percentage = (count / total) * 100
    deviation = ((count - avg_count) / avg_count) * 100
    
    # Color coding
    if deviation > 20:
        status = "⬆️ "
    elif deviation < -20:
        status = "⬇️ "
    else:
        status = "✓ "
    
    print(f"{status}{class_name:8s}: {bar:40s} {count:4d} ({percentage:5.1f}%) [{deviation:+.0f}%]")

print("-" * 80)

# Recommendations
print("\n💡 RECOMMENDATIONS:")

if max_count / min_count > 3:
    print("❌ SEVERE IMBALANCE detected!")
    print("   → Use class weights during training (already included in train_balanced.py)")
    print("   → Consider augmenting underrepresented classes")
elif max_count / min_count > 2:
    print("⚠️  MODERATE IMBALANCE detected")
    print("   → Class weights will help (included in train_balanced.py)")
else:
    print("✅ Dataset is reasonably balanced")

# Check for very small classes
small_classes = [name for name, count in class_counts.items() if count < 50]
if small_classes:
    print(f"\n⚠️  Classes with <50 images: {', '.join(small_classes)}")
    print("   → Consider collecting more data for these classes")

print("\n" + "=" * 80)