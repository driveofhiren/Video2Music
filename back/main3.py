import cv2
from ultralytics import YOLO

def main():
    # Load YOLO model
    yolo_model = YOLO('yolov8l.pt').cuda()  
    
    # Open camera stream
    cap = cv2.VideoCapture('http://192.168.2.106:8080/video')
    
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for consistent processing
        frame_resized = cv2.resize(frame, (1280, 720))
        
        # Perform object detection only
        results = yolo_model(frame_resized, verbose=False)[0]
        
        # Draw bounding boxes
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
            
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("YOLO Object Detection", frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import torch  # Added import that was missing
    main()