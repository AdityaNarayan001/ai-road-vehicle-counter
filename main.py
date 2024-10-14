from ultralytics import YOLO
import cv2 
import numpy as np

model_path = "./model/yolov5su.pt"

## Use resized video, Perform resizing using 'resizing-video.py' from utils directory.
video_path = "./asset/resized_video.mp4"

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

skip_frames = 5
side_panel = 250

incoming_ids = []
total_incoming_ids = set({})
incoming_car = []
incoming_bus_truck = []

outgoing_ids = []
total_outgoing_ids = set({})
outgoing_car = []
outgoing_bus_truck = []

diversing_ids = []
total_diversing_ids = set({})
diverging_car = []
diverging_bus_truck = []

vehicle_counter = {}

in_start_point = (573, 430) 
in_end_point = (856, 430)   
in_color = (0, 255, 0)      
in_thickness = 2 

out_start_point = (345, 195) 
out_end_point = (470, 195)   
out_color = (247, 145, 82)      
out_thickness = 2 

div_start_point = (190, 200) 
div_end_point = (335, 200)   
div_color = (0, 0, 255)      
div_thickness = 2 

if not cap.isOpened():
    print("Error : Could not open video !")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of Video")
        break

    results = model.track(frame, persist=True, verbose=False)
    if results is not None:
        for result in results:
            if result.boxes is not None:
                if hasattr(result.boxes, 'xyxy') and hasattr(result.boxes, 'id'):
                    boxes = result.boxes.xyxy.cpu().numpy()  
                    scores = result.boxes.conf.cpu().numpy()  
                    class_ids = result.boxes.cls.cpu().numpy()
                    try:
                        ids = result.boxes.id.cpu().numpy()

                        # Total Vehicle in frame count
                        vehicle_counter['total-vehicle'] = len(result.boxes.id)
                    except:
                        pass

                    try:
                        for box, score, class_id, detection_id in zip(boxes, scores, class_ids, ids):
                            x1, y1, x2, y2 = map(int, box)
                            label = f"{model.names[int(class_id)]} {score:.2f} ID:{int(detection_id)}"

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 1)
                            
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            label_y = max(y1, label_size[1] + 10)
                            cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y + 5), (0,0,0), cv2.FILLED)
                            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

                            # Incoming Traffic Counter
                            if center_x >= in_start_point[0]-50 and center_y <= in_end_point[1] and (class_id == 5 or class_id == 7):
                                incoming_bus_truck.append(0)
                            if center_x >= in_start_point[0]-50 and center_y <= in_end_point[1] and class_id == 2:
                                incoming_ids.append(detection_id)
                                incoming_car.append(0)
                                for i in incoming_ids:
                                    total_incoming_ids.add(i)
                                    

                            # Outgoing Traffic Counter
                            if center_x >= in_start_point[0]-50 and center_y <= in_end_point[1] and (class_id == 5 or class_id == 7):
                                outgoing_bus_truck.append(0)
                            if center_x <= in_start_point[0]-50 and center_x >= out_start_point[0] and center_y <= out_start_point[1]+100 and class_id == 2:
                                outgoing_car.append(0)
                                outgoing_ids.append(detection_id)
                                for i in outgoing_ids:
                                    total_outgoing_ids.add(i)
                                    

                            # Diverging Traffic Counter
                            if center_x >= in_start_point[0]-50 and center_y <= in_end_point[1] and (class_id == 5 or class_id == 7):
                                diverging_bus_truck.append(0)
                            if center_x <= div_end_point[0] and center_y <= div_end_point[1]+80 and class_id == 2:
                                diverging_car.append(0)
                                diversing_ids.append(detection_id)
                                for i in diversing_ids:
                                    total_diversing_ids.add(i)
                                    
                    except:
                        pass

    ## VIDEO FEED MARKERS
    cv2.rectangle(frame, (0, 0), (280, 40), (0,0,0), cv2.FILLED)
    cv2.putText(frame, f"Total Vehicles in Frame : {vehicle_counter['total-vehicle']}", (10, 23), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 1)

    cv2.line(frame, in_start_point, in_end_point, in_color, in_thickness)
    cv2.line(frame, in_start_point, (in_start_point[0]-50, 200), in_color, in_thickness)
    cv2.putText(frame, "Incoming", (in_start_point[0], in_start_point[1]-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, in_color, 1)

    cv2.line(frame, out_start_point, out_end_point, out_color, out_thickness)
    cv2.line(frame, out_start_point, (out_end_point[1]-120, 400), out_color, out_thickness)
    cv2.putText(frame, "Outgoing", (out_start_point[0], out_start_point[1]-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, out_color, 1)

    cv2.line(frame, div_start_point, div_end_point, div_color, div_thickness)
    cv2.putText(frame, "Diverting", (div_start_point[0], div_start_point[1]-5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, div_color, 1)

    canvas_height = frame.shape[0]
    canvas_width = frame.shape[1] + (2 * side_panel)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype="uint8")
    canvas[:, side_panel:side_panel + frame.shape[1]] = frame

    ## RIGHT PANEL
    cv2.rectangle(canvas, (side_panel+frame.shape[1]+10, 10), (frame.shape[1]+side_panel+side_panel-10, 464), (255,255,255), cv2.FILLED)
    cv2.putText(canvas, "Traffic Control Based", (side_panel+frame.shape[1]+10, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "on Traffic Flow : ", (side_panel+frame.shape[1]+10, 45), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "Refer Traffic Map to", (side_panel+frame.shape[1]+10, 434), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1)
    cv2.putText(canvas, "understand Terminals.", (side_panel+frame.shape[1]+10, 454), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1)

    right_center_x = (canvas_width-((side_panel-10)//2))
    right_center_y = frame.shape[0]//2
    cv2.circle(canvas, (right_center_x, right_center_y), radius=40, color=(0, 0, 0), thickness=-1)
    cv2.circle(canvas, (right_center_x, right_center_y), radius=30, color=(255, 255, 255), thickness=-1)

    cv2.putText(canvas, "T1", (right_center_x-110, right_center_y+5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "T2", (right_center_x-10, right_center_y-90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "T3", (right_center_x-10, right_center_y+90), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, "T4", (right_center_x+80, right_center_y+5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)

    def T1(t1_flag):
        cv2.circle(canvas, (right_center_x-60, right_center_y), radius=12, color=(0, 0, 0), thickness=-1)
        if t1_flag:
            output = cv2.circle(canvas, (right_center_x-60, right_center_y), radius=10, color=(0, 255, 0), thickness=-1)
        else:
            output = cv2.circle(canvas, (right_center_x-60, right_center_y), radius=10, color=(0, 0, 255), thickness=-1)
        return output

    def T2(t2_flag):
        cv2.circle(canvas, (right_center_x, right_center_y-60), radius=12, color=(0, 0, 0), thickness=-1)
        if t2_flag:
            output = cv2.circle(canvas, (right_center_x, right_center_y-60), radius=10, color=(0, 255, 0), thickness=-1)
        else:
            output = cv2.circle(canvas, (right_center_x, right_center_y-60), radius=10, color=(0, 0, 255), thickness=-1)
        return output

    def T3(t3_flag):
        cv2.circle(canvas, (right_center_x, right_center_y+60), radius=12, color=(0, 0, 0), thickness=-1)
        if t3_flag:
            output = cv2.circle(canvas, (right_center_x, right_center_y+60), radius=10, color=(0, 255, 0), thickness=-1)
        else:
            output = cv2.circle(canvas, (right_center_x, right_center_y+60), radius=10, color=(0, 0, 255), thickness=-1)
        return output
    
    def T4():
        cv2.circle(canvas, (right_center_x+60, right_center_y), radius=12, color=(0, 0, 0), thickness=-1)
        cv2.circle(canvas, (right_center_x+60, right_center_y), radius=10, color=(0, 255, 0), thickness=-1)

    ## --------------- TRAFFIC LIGTH LOGIC ---------------
    if len(set(incoming_ids)) > len(set(outgoing_ids)):
        T1(False)
        T2(False)
        T3(True)
        T4()
    if len(set(incoming_ids)) > len(set(diversing_ids)):
        T1(False)
        T2(False)
        T3(True)
        T4()
    if len(set(diversing_ids)) > len(set(incoming_ids)):
        T1(True)
        T2(False)
        T3(False)
        T4()
    if len(set(diversing_ids)) > len(set(outgoing_ids)):
        T1(True)
        T2(False)
        T3(False)
        T4()
    if len(set(outgoing_ids)) > len(set(incoming_ids)):
        T1(False)
        T2(True)
        T3(False)
        T4()
    if len(set(outgoing_ids)) > len(set(diversing_ids)):
        T1(False)
        T2(True)
        T3(False)
        T4()
    ## ---------------------------------------------------

    ## LEFT PANEL
    cv2.rectangle(canvas, (10, 10), (240, 148), in_color, cv2.FILLED)
    cv2.putText(canvas, "Incoming Traffic Details :", (10, 23), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, f"Live Vehicle Count : {len(set(incoming_ids))}", (10, 43), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Total Vehicle Count : {len(total_incoming_ids)}", (10, 63), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Live Car Count : {len(incoming_car)}", (10, 83), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Live Bus/Truck Count : {len(incoming_bus_truck)}", (10, 103), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    incoming_ids = []
    incoming_car = []
    incoming_bus_truck = []

    cv2.rectangle(canvas, (10, 158), (240, 306), out_color, cv2.FILLED)
    cv2.putText(canvas, "Outgoing Traffic Details :", (10, 170), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, f"Live Vehicle Count : {len(set(outgoing_ids))}", (10, 191), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Total Vehicle Count : {len(total_outgoing_ids)}", (10, 212), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Live Car Count : {len(outgoing_car)}", (10, 233), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Live Bus/Truck Count : {len(outgoing_bus_truck)}", (10, 254), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    outgoing_ids = []
    outgoing_car = []
    outgoing_bus_truck = []

    cv2.rectangle(canvas, (10, 316), (240, 464), div_color, cv2.FILLED)
    cv2.putText(canvas, "Diverting Traffic Details :", (10, 329), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)
    cv2.putText(canvas, f"Live Vehicle Count : {len(set(diversing_ids))}", (10, 349), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Total Vehicle Count : {len(total_diversing_ids)}", (10, 369), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Total Car Count : {len(diverging_car)}", (10, 389), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    cv2.putText(canvas, f"Total Bus/Truck Count : {len(diverging_bus_truck)}", (10, 409), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
    diversing_ids = []
    diverging_car = []
    diverging_bus_truck = []

    

    cv2.imshow('Video', canvas)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    for _ in range(skip_frames):
        cap.grab() 


cap.release()
cv2.destroyAllWindows()