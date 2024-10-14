from ultralytics import SAM
import cv2
import numpy as np

image_path = "asset/img.png"

model = SAM("model/sam_b.pt")
image = cv2.imread(image_path)

image_height, image_width = image.shape[:2]

results = model(image)
for res in results:
        masks = res.masks

if masks is not None:
    mask_array = masks.data.cpu().numpy()
    combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for mask in mask_array:
        mask = mask.astype(np.uint8)
        resized_mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.maximum(combined_mask, resized_mask)

    seg_frame_colored = cv2.applyColorMap((combined_mask * 255).astype(np.uint8), cv2.COLORMAP_COOL)
    blended_frame = cv2.addWeighted(image, 0.7, seg_frame_colored, 0.3, 0)

cv2.imshow('OG Video', image)
cv2.imshow('Segmented',seg_frame_colored)
cv2.imshow('Blended', blended_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
