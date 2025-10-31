import os
import shutil
import random
import tensorflow as tf
from tqdm import tqdm

# --- Config ---
ORIGINAL_DIR = "TB_Chest_Radiography_Database"
OUTPUT_DIR = "TB_Chest_Radiography_Database_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 27
SPLIT_RATIO = [0.8, 0.10, 0.10]  # train, val, test

# --- GPU Setup ---
def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


# --- Split Dataset (Run Once) ---
def split_dataset():
    if os.path.exists(OUTPUT_DIR):
        print("Dataset already split.")
        return

    print("Splitting dataset into train/val/test...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in ["Normal", "Tuberculosis"]:
        files = os.listdir(os.path.join(ORIGINAL_DIR, cls))
        random.seed(SEED)
        random.shuffle(files)

        total = len(files)
        train_end = int(total * SPLIT_RATIO[0])
        val_end = train_end + int(total * SPLIT_RATIO[1])

        split_files = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:]
        }

        for split in split_files:
            split_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for file in tqdm(split_files[split], desc=f"{cls} - {split}"):
                shutil.copy2(os.path.join(ORIGINAL_DIR, cls, file), os.path.join(split_dir, file))


# --- Main Pipeline ---
def main():
    setup_gpu() # Setup GPU memory growth
    split_dataset()

if __name__ == "__main__":
    main()
