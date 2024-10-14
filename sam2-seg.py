from ultralytics import SAM
import cv2
import numpy as np

video_path = "asset/resized_video.mp4"
model = SAM("model/sam2_b.pt")

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
                mask = mask.astype(np.uint8)
                resized_mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                combined_mask = np.maximum(combined_mask, resized_mask)

            seg_frame_colored = cv2.applyColorMap((combined_mask * 255).astype(np.uint8), cv2.COLORMAP_COOL)
            blended_frame = cv2.addWeighted(frame, 0.7, seg_frame_colored, 0.3, 0)


    cv2.imshow('OG Video', frame)
    cv2.imshow('Segmented',seg_frame_colored)
    cv2.imshow('Blended', blended_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
