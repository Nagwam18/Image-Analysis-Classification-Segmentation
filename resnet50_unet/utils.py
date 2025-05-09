# resnet50_unet/utils.py

import os
import zipfile
import urllib.request
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def download_and_extract_dataset(url, dest_dir="PetImages"):
    """Downloads and extracts the Cats & Dogs dataset if not already present."""
    zip_path = "cats_dogs.zip"

    if os.path.exists(dest_dir):
        print(f"Dataset already exists at '{dest_dir}'. Skipping download.")
        return

    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("Extraction complete.")

    # Clean up the zip file after extraction
    os.remove(zip_path)
    print(f"Removed downloaded zip file: {zip_path}")

def delete_corrupted_images(data_dir="PetImages"):
    """Deletes corrupted images in the dataset folder."""
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(data_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "rb") as f:
                if b"JFIF" not in f.peek(10):
                    os.remove(fpath)
                    num_skipped += 1
    print(f"Deleted {num_skipped} corrupted images.")

def load_cat_dog_data(data_dir="PetImages", batch_size=32, image_size=(180, 180), seed=123):
    """Loads cat and dog data for training and validation."""
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size
    )

    return train_ds, val_ds

def preprocess_input(image_batch):
    """Preprocess input images for ResNet50."""
    return resnet_preprocess(image_batch)
