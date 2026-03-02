import os
import glob

# Point to the specific broken folder
folder_path = "/scratch/lt200353-pcllm/location/real_colon/001-012_frames"

# Find all files ending in .0.jpg
bad_files = glob.glob(os.path.join(folder_path, "*.0.jpg"))
print(f"Found {len(bad_files)} files to fix...")

for old_path in bad_files:
    # Safely replace the exact problematic string at the end of the file
    new_path = old_path.replace(".0.jpg", ".jpg")
    os.rename(old_path, new_path)

print("Done! All files renamed.")
