from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from ultralytics import YOLO
import os, shutil, zipfile, uuid, cv2

app = FastAPI()
model = YOLO('yolov8n.pt')
coco_classes = model.names

@app.post("/annotate/")
async def annotate_images(
    files: list[UploadFile] = File(...),
    label: str = Form(...)
):
    if label not in coco_classes.values():
        return {"error": f"'{label}' not in YOLOv8 COCO class list."}

    # Setup session folders
    session_id = str(uuid.uuid4())
    base_dir = f"temp/{session_id}"
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    annotated_dir = os.path.join(base_dir, "annotated")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    target_class_id = [k for k, v in coco_classes.items() if v == label][0]

    # Process each image
    for file in files:
        filename = file.filename
        image_path = os.path.join(images_dir, filename)

        with open(image_path, "wb") as f:
            f.write(await file.read())

        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        results = model(image_path)[0]
        label_lines = []

        # Convert BGR to RGB for drawing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for box in results.boxes:
            cls_id = int(box.cls)
            if cls_id != target_class_id:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw rectangle
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # YOLO format: class_id x_center y_center width height (normalized)
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            label_lines.append(f"{target_class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # Save label file
        if label_lines:
            label_file_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
            with open(label_file_path, "w") as f:
                f.write("\n".join(label_lines))

        # Save annotated image
        annotated_path = os.path.join(annotated_dir, filename)
        cv2.imwrite(annotated_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Create ZIP file with both labels and annotated images
    zip_path = os.path.join(base_dir, "results.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for folder_name in ['labels', 'annotated']:
            folder_path = os.path.join(base_dir, folder_name)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, base_dir)
                    zipf.write(file_path, arcname)

    return FileResponse(zip_path, filename="results.zip", media_type="application/zip")
