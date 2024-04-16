import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.mask import decode
from pprint import pprint
import urllib.request

use_offline = True

dataDir = "/mnt/x/dataset/coco2017"
dataType = "train2017"
annFile = f"{dataDir}/raw/instances_{dataType}.json"

if use_offline:
    imagesDir = f"{dataDir}/train/data"

outPath = "./out"

coco = COCO(annFile)

catIds = coco.getCatIds(catNms=["truck"])

print("Number of images with trucks: ", len(coco.getImgIds(catIds=catIds)))

imgIds = coco.getImgIds(catIds=catIds)


if not os.path.exists(f"{outPath}/images"):
    os.makedirs(f"{outPath}/images")

if not os.path.exists(f"{outPath}/labels"):
    os.makedirs(f"{outPath}/labels")

def get_poly(mask):
    if isinstance(mask, list):
        _poly = np.array(mask).reshape(-1, 2).astype(np.int32)
    return _poly

def draw_mask(img, poly, id=None):
    img = cv2.polylines(img, [poly], True, (255, 0, 255), 2)
    if id is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (poly[0][0], poly[0][1])
        fontScale = 0.7
        fontColor = (255, 0, 0)
        lineType = 1
        cv2.putText(img, str(id), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    return img

def poly2yolo(poly, img_shape):
    if poly is not None and img_shape is not None:
        if len(poly) > 2 and img_shape[0] > 0 and img_shape[1] > 0:
            yolo_poly = []
            for point in poly:
                x = point[0] / img_shape[1]
                y = point[1] / img_shape[0]
                yolo_poly.extend([x, y])
            return yolo_poly

count = len(imgIds)

# start image index
i = 0
class_id = 0
saved_ids = []

# Получить список сохраненных изображений
if os.path.exists(outPath):
    saved_images = os.listdir(f"{outPath}/images")
    for saved_image in saved_images:
        saved_ids.append(imgIds.index(int(saved_image.split(".")[0])))

while i < count:
    current_masks = []
    # print(f"Image {i+1}/{len(imgIds)}")
    img = coco.loadImgs(imgIds[i])[0]
    file_name = ".".join(img["file_name"].split(".")[:-1])
    if use_offline:
        I = cv2.imread(f"{imagesDir}/{file_name}.jpg")
    else:
        url = img["coco_url"]
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        I = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img_shape = I.shape[:2]
    annIds = coco.getAnnIds(imgIds=img["id"], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for n, seg in enumerate(anns):
        mask = seg["segmentation"][0]
        poly = get_poly(mask)
        I = draw_mask(I, poly, n+1)
        yolo_poly = poly2yolo(poly, img_shape)
        if yolo_poly is not None:
            current_masks.append(yolo_poly)

    render = I.copy()

    if i in saved_ids:
        # Рисукем желтую рамку вокруг изображения, если оно уже сохранено
        render = cv2.rectangle(render, (0, 0), (render.shape[1], render.shape[0]), (0, 255, 255), 10)

    title = f"Image {i+1}/{len(imgIds)} - {file_name} - {len(anns)} trucks - {len(annIds)} annotations"
    cv2.setWindowTitle("Image", title)
    cv2.imshow("Image", cv2.resize(render, (1600, 1200)))
    key = cv2.waitKey(0) & 0xFF
    match key:
        # <esc>
        case 27:
            break
        # <right>
        case 83:
            # Проверяем, что это не последний элемент
            if i < count - 1:
                i += 1
        # <left>
        case 81:
            # Проверяем, что это не первый элемент
            if i > 0:
                i -= 1
        # <down>
        case 84:
            # Сохраняем текущее изображение в images
            cv2.imwrite(f"{outPath}/images/{file_name}.jpg", I)
            # Сохраняем текущие маски в labels
            with open(f"{outPath}/labels/{file_name}.txt", "w") as f:
                for mask in current_masks:
                    f.write(f"{class_id} {' '.join(map(str, mask))}\n")
                print(f"Frame {file_name} saved, {len(current_masks)} masks")
                saved_ids.append(i)

        case 82:
            # Удаляем текущее изображение из images
            if os.path.exists(f"{outPath}/images/{file_name}.jpg"):
                os.remove(f"{outPath}/images/{file_name}.jpg")
            # Удаляем текущие маски из labels
            if os.path.exists(f"{outPath}/labels/{file_name}.txt"):
                os.remove(f"{outPath}/labels/{file_name}.txt")
            print(f"Frame {file_name} removed")
            saved_ids.remove(i)

cv2.destroyAllWindows()
