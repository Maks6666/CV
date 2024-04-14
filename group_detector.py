import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("path/to/video")

ret, frame = cap.read()
object_count = 0

model = YOLO("yolov8n")
names = model.names
threshold = 0.5

while ret:

    person_coordinates = []
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        class_name = names[int(class_id)].upper()

        if score > threshold and class_name == "PERSON":
            person_coordinates.append((x1, y1))
            person_coordinates.append((x2, y2))



    if person_coordinates:
        x_min = min(coord[0] for coord in person_coordinates)
        y_min = min(coord[1] for coord in person_coordinates)
        x_max = max(coord[0] for coord in person_coordinates)
        y_max = max(coord[1] for coord in person_coordinates)

        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)

        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
        cv2.circle(frame, (x_center, y_center + 80), 3, (0, 0, 255), thickness=cv2.FILLED)
        cv2.putText(frame, f"{names[int(class_id)].upper()}", (int(x_min), int(y_min)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Video", frame)

    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

