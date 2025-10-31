import os
import pydicom
import numpy as np
from PIL import Image
import re

def get_next_index_tb(target_folder, prefix="TB"):
    existing = [f for f in os.listdir(target_folder) if f.lower().endswith(".png") and f.startswith(prefix)]
    numbers = [int(re.findall(rf"{prefix}(\d+)\.png", f)[0]) for f in existing if re.findall(rf"{prefix}(\d+)\.png", f)]
    return max(numbers, default=0) + 1

def convert_dicoms_to_png(source_folder, target_tb_folder, prefix="TB-"):
    os.makedirs(target_tb_folder, exist_ok=True)
    index = get_next_index_tb(target_tb_folder, prefix=prefix)

    for filename in sorted(os.listdir(source_folder)):
        if filename.lower().endswith((".dcm", ".dicom")):
            dicom_path = os.path.join(source_folder, filename)
            output_filename = f"{prefix}{index:05d}.png"
            output_path = os.path.join(target_tb_folder, output_filename)

            try:
                ds = pydicom.dcmread(dicom_path)
                pixel_array = ds.pixel_array.astype(np.float32)

                # Apply rescale slope/intercept if present
                intercept = ds.get("RescaleIntercept", 0.0)
                slope = ds.get("RescaleSlope", 1.0)
                pixel_array = pixel_array * slope + intercept

                # Normalize to 0–255
                pixel_array -= pixel_array.min()
                pixel_array /= pixel_array.max()
                pixel_array = (pixel_array * 255).astype(np.uint8)

                # Convert grayscale to RGB if needed
                if len(pixel_array.shape) == 2:
                    pixel_array = np.stack([pixel_array] * 3, axis=-1)

                Image.fromarray(pixel_array).save(output_path)
                print(f"Saved {output_filename}")
                index += 1
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    # Example usage: loop over multiple folders
    source_folders_tb = ["intbtr251","intbtr252","intbtr253","intbtr254", "intbtr255","intbtr256","intbtr260","intbtr261"]  # Add more as needed
    target_tb_folder = os.path.join("TB_Chest_Radiography_Database_raw", "TB")

    for folder in source_folders_tb:
        print(f"🔍 Processing folder: {folder}")
        convert_dicoms_to_png(folder, target_tb_folder, prefix="TB-")
 
print("TB Conversion complete!")

print("Starting Normal dicom files conversion to PNG...")


def get_next_index_norm(target_folder, prefix="Normal"):
    existing = [f for f in os.listdir(target_folder) if f.lower().endswith(".png") and f.startswith(prefix)]
    numbers = [int(re.findall(rf"{prefix}(\d+)\.png", f)[0]) for f in existing if re.findall(rf"{prefix}(\d+)\.png", f)]
    return max(numbers, default=0) + 1

def convert_dicoms_to_png(source_folder, target_norm_folder, prefix="Normal-"):
    os.makedirs(target_norm_folder, exist_ok=True)
    index = get_next_index_norm(target_norm_folder, prefix=prefix)

    for filename in sorted(os.listdir(source_folder)):
        if filename.lower().endswith((".dcm", ".dicom")):
            dicom_path = os.path.join(source_folder, filename)
            output_filename = f"{prefix}{index:05d}.png"
            output_path = os.path.join(target_norm_folder, output_filename)

            try:
                ds = pydicom.dcmread(dicom_path)
                pixel_array = ds.pixel_array.astype(np.float32)

                # Apply rescale slope/intercept if present
                intercept = ds.get("RescaleIntercept", 0.0)
                slope = ds.get("RescaleSlope", 1.0)
                pixel_array = pixel_array * slope + intercept

                # Normalize to 0–255
                pixel_array -= pixel_array.min()
                pixel_array /= pixel_array.max()
                pixel_array = (pixel_array * 255).astype(np.uint8)

                # Convert grayscale to RGB if needed
                if len(pixel_array.shape) == 2:
                    pixel_array = np.stack([pixel_array] * 3, axis=-1)

                Image.fromarray(pixel_array).save(output_path)
                print(f"Saved {output_filename}")
                index += 1
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

if __name__ == "__main__":
    # Example usage: loop over multiple folders
    source_folders_norm = ["intbtr12","intbtr13","intbtr29","intbtr65","intbtr67","intbtr78","intbtr79","intbtr81","intbtr103","intbtr104"]  # Add more as needed
    target_norm_folder = os.path.join("TB_Chest_Radiography_Database_raw", "Normal")

    for folder in source_folders_norm:
        print(f"🔍 Processing folder: {folder}")
        convert_dicoms_to_png(folder, target_norm_folder, prefix="Normal-")
 