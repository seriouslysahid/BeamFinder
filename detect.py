import argparse
import csv
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="BeamFinder - detect drones in images")
    p.add_argument("--model", default="runs/drone_detect/weights/best.pt",
                    help="path to trained model weights")
    p.add_argument("--images", default="data/images/validation",
                    help="folder of images to run detection on")
    p.add_argument("--conf", type=float, default=0.4,
                    help="min confidence to count as a detection")
    p.add_argument("--device", default="0", help="GPU id or 'cpu'")
    p.add_argument("--imgsz", type=int, default=960,
                    help="inference resolution (should match training)")
    p.add_argument("--export", choices=["onnx", "engine"], default=None,
                    help="export model to ONNX or TensorRT format, then exit")
    p.add_argument("--sahi", action="store_true",
                    help="enable SAHI sliced inference (better for small drones)")
    return p.parse_args()


def write_detection(writer, image_name, x1, y1, w, h, conf, cls_name):
    cx = x1 + w / 2
    cy = y1 + h / 2
    writer.writerow([
        image_name,
        round(x1, 2), round(y1, 2), round(w, 2), round(h, 2),
        round(cx, 2), round(cy, 2),
        conf, cls_name
    ])


def run_sahi_inference(img_path, sahi_model, get_sliced_prediction):
    result = get_sliced_prediction(
        str(img_path), sahi_model,
        slice_height=512, slice_width=512,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        verbose=0,
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1 = bbox.minx, bbox.miny
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny
        detections.append((x1, y1, w, h, round(pred.score.value, 4),
                           pred.category.name))
    return detections, result


def run_yolo_inference(img_path, model, args):
    result = model(
        str(img_path),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        half=True,
        verbose=False,
    )[0]

    detections = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cls_id = int(boxes.cls[i].item())
            detections.append((x1, y1, x2 - x1, y2 - y1,
                               round(boxes.conf[i].item(), 4),
                               result.names[cls_id]))
    return detections, result


if __name__ == "__main__":
    args = parse_args()

    image_dir = Path(args.images)
    output_csv = Path("output/detections.csv")
    output_images = Path("output/annotated")

    if not image_dir.exists():
        print(f"Error: image directory '{image_dir}' not found.")
        raise SystemExit(1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  BeamFinder - Drone Detection")
    print("=" * 55)
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Conf:   {args.conf}")
    print(f"  ImgSz:  {args.imgsz}px")
    print(f"  SAHI:   {'on' if args.sahi else 'off'}")
    print(f"  Input:  {image_dir}")
    print(f"  Output: {output_csv}")
    print("=" * 55)
    print()

    model = YOLO(args.model)

    if args.export:
        model.export(format=args.export, imgsz=args.imgsz, half=True)
        print(f"Exported to {args.export} format.")
        raise SystemExit(0)

    sahi_model = None
    get_sliced_prediction = None
    if args.sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=args.model,
            confidence_threshold=args.conf,
            device=f"cuda:{args.device}" if args.device != "cpu" else "cpu",
            image_size=args.imgsz,
        )

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    total_detections = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "x", "y", "width", "height",
                         "cx", "cy", "confidence", "class"])

        for i, img_path in enumerate(image_paths, 1):
            name = img_path.name

            if sahi_model:
                dets, sahi_result = run_sahi_inference(
                    img_path, sahi_model, get_sliced_prediction)
                sahi_result.export_visuals(
                    export_dir=str(output_images), file_name=img_path.stem)
            else:
                dets, yolo_result = run_yolo_inference(img_path, model, args)
                yolo_result.save(filename=str(output_images / name))

            if not dets:
                writer.writerow([name, "", "", "", "", "", "", "", ""])
                print(f"  [{i}] {name} - no detections")
                continue

            total_detections += len(dets)
            for x1, y1, w, h, conf, cls_name in dets:
                write_detection(writer, name, x1, y1, w, h, conf, cls_name)
            print(f"  [{i}] {name} - {len(dets)} detection(s)")

    print()
    print("=" * 55)
    print(f"  Processed {len(image_paths)} images, {total_detections} total detections")
    print(f"  Results: {output_csv}")
    print("=" * 55)
