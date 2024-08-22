import cv2
from ultralytics import YOLO

model = YOLO('maskDetector.pt')

video_path = 'test/video/mask_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
out = cv2.VideoWriter('mask_detection_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply YOLOv8 to the frame
    results = model(frame)

    # Visualize the results
    annotated_frame = results[0].plot()

    # Write the frame with detections to the output video
    out.write(annotated_frame)

    # Optionally display the frame with detections
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()