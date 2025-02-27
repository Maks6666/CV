from deepsort import tracker
from group_detector import model
from sort import Sort
from ultralytics import YOLO
import numpy as np
import cv2
import torch

class SortTracker:
    def __init__(self, path, device, yolo):
        self.path = path
        self.device = device
        self.yolo = yolo
        self.model = self.load_model()
        self.names = self.model.names
        self.sort = Sort(max_age=50, min_hits=8, iou_threshold=0.4)

    def load_model(self):
        model = YOLO(self.yolo)
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        results = model(frame)[0]
        return results

    def get_results(self, results):
        res_arr = []
        for result in results[0]:
            bboxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_id = result.boxes.cls.cpu().numpy()

            arr = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], scores[0], class_id[0]]
            res_arr.append(arr)

        return np.array(res_arr)

    def draw(self, frame, bboxes, idc, clss):
        for bbox, idx, cls in zip(bboxes, idc, clss):
            x1, y1, x2, y2 = bbox
            name = self.names[int(cls)]
            text = f"{idx}:{name}"

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame)
            results_array = self.get_results(results)

            if len(results_array) == 0:
                results_array = np.empty((0, 5))

            res = self.sort.update(results_array)

            bboxes = res[:, :-1]
            idc = res[:, -1].astype(int)
            clss = results_array[:, -1].astype(int)

            upd_frame = self.draw(frame, bboxes, idc, clss)
            cv2.imshow('Video', upd_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

path = 1
device = "mps" if torch.backends.mps.is_available() else "cpu"
# you may use any YOLO model here
yolo = "yolo11.pt"
tracker = SortTracker(path, device, yolo)
tracker()


