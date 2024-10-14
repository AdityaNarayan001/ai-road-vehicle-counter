import cv2

start_point = None
end_point = None
drawing = False 


def draw_line(event, x, y, flags, param):
    global start_point, end_point, drawing
    
    # When left mouse button is pressed down, record the starting point
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True
        print('Start Point : ',start_point)

    # When the mouse is moved, record the end point if drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    # When left mouse button is released, stop drawing
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        print('End Point : ',end_point)
        # Draw the final line
        if start_point and end_point:
            cv2.line(frame, start_point, end_point, (0, 255, 0), 5)

video_path = "/Users/aditya.narayan/Desktop/Road-Vehicle-Counter/asset/resized_video.mp4"
cap = cv2.VideoCapture(video_path)


if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", draw_line)


while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    if start_point and end_point:
        cv2.line(frame, start_point, end_point, (0, 255, 0), 5)

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
