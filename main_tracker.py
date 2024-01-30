import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import random

from sort import Sort

class ObjectDetection:

    def __init__(self, capture_index):
        # Initialize ObjectDetection class with the given capture index for video feed
        self.capture_index = capture_index

        # Check if CUDA is available, use 'cuda' if available, otherwise use 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device: ', self.device)

        # Load YOLOv8 model
        self.model = self.load_model()

        # Create a dictionary mapping class indices to class names
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        # Load pretrained YOLOv8 model
        model = YOLO('yolov8m.pt')
        model.fuse()

        return model
    
    def predict(self, frame):
        # Perform object detection on the given frame using the YOLO model
        results = self.model(frame, verbose=False)
        return results
    
    def get_results(self, results):
        # Extract relevant information (class ID, bounding box, confidence) from YOLO results
        detections_list = []

        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if (class_id == 0):
                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()

                # Merge bounding box coordinates and confidence into a single detection entry
                merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0]]

                detections_list.append(merged_detection)
        
        return np.array(detections_list)
    
    def draw_bboxes_with_id(self, img, bboxes, ids):
        # Draw bounding boxes with associated IDs on the image
        for bbox, id in zip(bboxes, ids):
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(img, 'ID: ' + str(id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img
    
    def draw_bounding_boxes_without_id(self, frame, results):
        # Draw bounding boxes on the frame without associated IDs
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, clss in zip(boxes, classes):
            # Generate a random color for each object based on its ID
            if clss != 0:
                random.seed(int(clss))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                cv2.putText(
                    frame,
                    f"{self.CLASS_NAMES_DICT[clss]}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
        return frame
    
    def __call__(self):
        # Open video capture with the specified index
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Initialize SORT tracker
        sort = Sort(max_age=200, min_hits=8, iou_threshold=0.5)

        while True:
            start_time = time()

            # Read a frame from the video feed
            ret, frame = cap.read()
            assert ret

            # Perform object detection on the frame
            results = self.predict(frame)
            # Extract relevant information from YOLO results
            detections_list = self.get_results(results)

            # SORT tracking
            if len(detections_list) == 0:
                detections_list = np.empty((0, 5))

            res = sort.update(detections_list)

            boxes_track = res[:,:-1]
            boxes_id = res[:,-1].astype(int)

            # Draw bounding boxes with associated IDs
            frame = self.draw_bboxes_with_id(frame, boxes_track, boxes_id)
            # Draw bounding boxes without associated IDs
            frame = self.draw_bounding_boxes_without_id(frame, results)

            end_time = time()

            # Calculate and display frames per second (FPS)
            fps = 1/np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            # Display the frame with bounding boxes
            cv2.imshow('YOLOv8 Detection', frame)
 
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Release the video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

# Create an instance of the ObjectDetection class with the specified capture index and execute it
detector = ObjectDetection(capture_index=0)
detector()