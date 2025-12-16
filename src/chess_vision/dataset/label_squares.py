import cv2
import os
import shutil

source_dir = "data"
empty_dir = os.path.join(source_dir, "empty")
occupied_dir = os.path.join(source_dir, "occupied")
os.makedirs(empty_dir, exist_ok=True)
os.makedirs(occupied_dir, exist_ok=True)

images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg'))]
images.sort()

print("Labeling started. Press 'e' for empty, 'o' for occupied, 'q' to quit.")

for img_file in images:
    img_path = os.path.join(source_dir, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_file}, skipping.")
        continue

    cv2.imshow("Label Image", img)
    key = cv2.waitKey(0) & 0xFF

    if key == 81:
        shutil.move(img_path, os.path.join(empty_dir, img_file))
        print(f"{img_file} -> empty")
    elif key == 83:
        shutil.move(img_path, os.path.join(occupied_dir, img_file))
        print(f"{img_file} -> occupied")
    elif key == ord('q'):
        print("Quitting labeling.")
        break
    else:
        print("Invalid key, skipping image.")

cv2.destroyAllWindows()
