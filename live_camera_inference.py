import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11l-seg.pt")
names = model.model.names

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        boxes = results[0].boxes.xyxy.cpu().tolist()

        for mask, cls, box in zip(masks, clss, boxes):
            color = colors(int(cls), True)
            txt_color = annotator.get_txt_color(color)

            overlay = im0.copy()
            cv2.fillPoly(overlay, [mask.astype(int)], color=color)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0, im0)

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, thickness=2)

            label = f"{names[int(cls)]}"
            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, thickness=2)

    cv2.imshow("Live Instance Segmentation", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
