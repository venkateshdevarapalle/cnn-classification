import os
import pandas as pd

def load_image_paths(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    data = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        label = folder.lower()  # Cat -> cat, Dog -> dog

        for img in os.listdir(folder_path):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                data.append({
                    "filename": os.path.join(folder_path, img),
                    "class": label
                })

    if len(data) == 0:
        raise RuntimeError("No images found. Check dataset structure.")

    df = pd.DataFrame(data)
    print(f"✅ Total images loaded: {len(df)}")
    print("Classes found:", df["class"].unique())
    return df
