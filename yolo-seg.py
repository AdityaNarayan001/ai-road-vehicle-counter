from ultralytics import YOLO
import cv2
import numpy as np

## Use resized video, Perform resizing using 'resizing-video.py' from utils directory.
video_path = "asset/resized_video.mp4"
model = YOLO("model/yolov8n-seg.pt")

skip_frames = 5

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error : Could not open video !")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of Video")
        break

    frame_height, frame_width = frame.shape[:2]

    results = model(frame)

    for res in results:
        masks = res.masks

        if masks is not None:
            mask_array = masks.data.cpu().numpy()
            combined_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            for mask in mask_array:
                resized_mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, resized_mask)

            seg_frame_colored = cv2.applyColorMap((combined_mask * 255).astype(np.uint8), cv2.COLORMAP_COOL)
            blended_frame = cv2.addWeighted(frame, 0.7, seg_frame_colored, 0.3, 0)


    cv2.imshow('OG Video', frame)
    cv2.imshow('Segmented',seg_frame_colored)
    cv2.imshow('Blended', blended_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    for _ in range(skip_frames):
        cap.grab() 


cap.release()
cv2.destroyAllWindows()
