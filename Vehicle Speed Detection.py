# #This is a Project on Vehicle Speed Detection System
# #Started on 29-08-2024
# #Core Libraries OPENCV(CV2) and NUMPY

# import numpy as n
# import cv2 as cv

# #main function-1
# #Capture Video
# def capture_video(Source=0):
#     rec=cv.VideoCapture(Source)
#     if not rec.isOpened():
#         print("Cannot open camera")
#         return None
    
#     print(f"Successfully connected to camera {Source}")
#     return rec
# def show_frames(choice):
#     back_subt=cv.createBackgroundSubtractorMOG2(history=250,varThreshold=20)
#     while choice ==1:
#         video_capture=capture_video("video/test3.mp4")
#         if video_capture is None:
#             print("Error:Could not capture video")
#             break
#         while True:
#             ret, frame = video_capture.read()
#             if not ret:
#                 print("Failed to grab frame or end of video.")
#                 break
            
#             # Resize the frame to a specific size, e.g., 640x480
#             resized_frame = cv.resize(frame, (900,650))
#             f1=apply_background_substraction(resized_frame,back_subt)

#             cv.imshow('Captured Frame', resized_frame)
#             cv.imshow("Masked Modified Omni",f1)

#             if cv.waitKey(1) & 0xFF == ord('q'):
#                 print("Video capture stopped by user.")
#                 break
#         # Release the capture and close the window
#         video_capture.release()
#         cv.destroyAllWindows()
#         print("Video capture released and windows closed.")

#         # Prompt user to capture video again or exit
#         choice = int(input("Enter 1 to capture video again, 2 to exit: "))
#     print("Program terminated.")

# #main function-2
# #apply background substraction
# def apply_background_substraction(frame,back_subt):
#     f2=back_subt.apply(frame)
#     kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
#     f2=cv.morphologyEx(f2,cv.MORPH_OPEN,kernel)
#     f2=cv.morphologyEx(f2,cv.MORPH_CLOSE,kernel)
#     return f2

# #Looping the program
# choice=int(input("Enter choice:"))
# if(choice==1):
#     show_frames(choice)
# elif(choice==2):
#     print("Exiting the program")
#     exit()
# else:
#     print("Wrong Choice")



import numpy as n
import cv2 as cv
from ultralytics import YOLO

# Load the YOLOv8 model (you can use 'yolov8n', 'yolov8s', 'yolov8m' for different model sizes)
model = YOLO('yolov8n.pt')
# #main function-1
# #Capture Video
def capture_video(Source=0):
    rec=cv.VideoCapture(Source)
    if not rec.isOpened():
        print("Cannot open camera")
        return None
    
    print(f"Successfully connected to camera {Source}")
    return rec
# main function-3
# Detect objects using YOLOv8
# def detect_objects(frame):
#     # Apply YOLOv8 model on the frame
#     results = model(frame)
#     # Extract bounding boxes, class names, and confidence scores
#     for result in results:
#         boxes = result.boxes.xyxy  # Box coordinates
#         confidences = result.boxes.conf  # Confidence score
#         class_ids = result.boxes.cls  # Class labels

#         # Loop through detected objects and draw bounding boxes
#         for box, conf, class_id in zip(boxes, confidences, class_ids):
#             x1, y1, x2, y2 = map(int, box)
#             label = f"{model.names[int(class_id)]}: {conf:.2f}"

#             # Draw the bounding box and label on the frame
#             cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame
# # #main function-2

# #apply background substraction
def apply_background_substraction(frame,back_subt):
    f2=back_subt.apply(frame)
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    f2=cv.morphologyEx(f2,cv.MORPH_OPEN,kernel)
    f2=cv.morphologyEx(f2,cv.MORPH_CLOSE,kernel)
    return f2

# Main function to capture video, apply background subtraction, and object detection
def show_frames(choice):
    back_subt = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=10)
    while choice == 1:
        video_capture = capture_video("video/test3.mp4")
        if video_capture is None:
            print("Error: Could not capture video")
            break
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame or end of video.")
                break

            # Resize the frame to a specific size, e.g., 640x480
            resized_frame = cv.resize(frame, (900, 650))
            f1 = apply_background_substraction(resized_frame, back_subt)
            
            # Detect objects on the background-subtracted frame
            # detection_frame = detect_objects(f1)

            cv.imshow('Captured Frame', resized_frame)
            cv.imshow("Masked Modified Omni", f1)
            # cv.imshow("Object Detection", detection_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Video capture stopped by user.")
                break

        # Release the capture and close the window
        video_capture.release()
        cv.destroyAllWindows()
        print("Video capture released and windows closed.")
        choice = int(input("Enter 1 to capture video again, 2 to exit: "))
    print("Program terminated.")

#Looping the program
choice=int(input("Enter choice:"))
if(choice==1):
    show_frames(choice)
elif(choice==2):
    print("Exiting the program")
    exit()
else:
    print("Wrong Choice")