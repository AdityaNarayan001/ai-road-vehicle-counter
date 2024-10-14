# Road Vehicle Counter and  Traffic Ligth Controller

### The repo contains 4 main files :
#### To interrupt-and-exit output window press ```q```.
⚫ ```main.py``` - This file uses ```YOLO - yolov5su.pt``` model to perform detction. The output video screenshot can be seen in ```./Road-Vehicle-Counter/images```. The output includes various values.
<br><br>
⚫ ```sam-seg-img.py``` - This file uses ```Segment Anything Model - sam_b.pt``` model by Meta. The file returns Segmented Objects in Image or Video. The file produces ```Original, Blended and Segmented``` output windows.
<br><br>
⚫ ```sam2-img.py``` - This file uses ```Segment Anything Model 2 - sam2_b.pt``` model by Meta. The file returns Segmented Objects in Image or Video. The file produces ```Original, Blended and Segmented``` output windows.
<br><br>
⚫ ```main.py``` - This file uses ```YOLO - yolov8n-seg.pt``` model to perform segmentation.

#### Use files in ```./Road-Vehicle-Counter/utils``` for ```resizing video```.