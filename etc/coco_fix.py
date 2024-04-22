from pathlib import Path
import shutil
import os
from tqdm import tqdm

# Путь к корневой папке датасета
dataDir = "/mnt/x/dataset/coco2017"
# Имя сплита (train2017, val2017)

imagesDir = Path(f"{dataDir}/train/data")
    
labels_out_dir = Path("./out/labels")
images_out_dir = Path("./out/images")

# получаем список имён файлов в labels_out_dir без расширения
labels_out_files = [f.stem for f in labels_out_dir.glob("*.txt")]

# Копируем из imagesDir в images_out_dir только те изображения, для которых есть файлы меток
for label_file in tqdm(labels_out_files, total=len(labels_out_files)):
    shutil.copy(imagesDir / Path(f"{label_file}.jpg"), images_out_dir / Path(f"{label_file}.jpg"))
