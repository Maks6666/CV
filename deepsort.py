from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
import cv2

class DeepTracker:
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.model = self.load_model()
        self.names = self.model.names
        self.tracker = DeepSort(max_iou_distance=0.4, max_age=50)

    def load_model(self):
        model = YOLO("yolo11n.pt")
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        results = self.model(frame)[0]
        return results


    def get_results(self, results, frame):
        res_array = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.5:
                res_array.append(([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], float(score), int(class_id)))

        tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            bboxes = track.to_tlbr()
            x1, y1, x2, y2 = bboxes
            idx = track.track_id
            class_id = track.get_det_class()
            name = self.names[int(class_id)]

            text = f"{idx}:{name}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame)
            upd_frame = self.get_results(results, frame)

            cv2.imshow('Deep Tracker', upd_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

path = 1
device = "mps" if torch.backends.mps.is_available() else "cpu"
tracker = DeepTracker(path, device)
tracker()