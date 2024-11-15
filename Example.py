#This is a Project on Vehicle Speed Detection System
#Started on 29-08-2024
#Core Libraries OPENCV(CV2) and NUMPY

import numpy as n
import cv2 as cv

#main function-1
#Capture Video
def capture_video(Source=0):
    rec=cv.VideoCapture(Source)
    if not rec.isOpened():
        print("Cannot open camera")
        return None
    
    print(f"Successfully connected to camera {Source}")
    return rec

def detect_vehicles(fg_mask, frame):
    # Find contours in the foreground mask
    contours, _ = cv.findContours(fg_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Minimum area threshold to filter small objects (you can adjust this value)
    min_area = 500

    # Loop through contours and draw bounding boxes around detected vehicles
    for contour in contours:
        if cv.contourArea(contour) > min_area:  # Check if the contour area is large enough
            # Get the bounding box coordinates
            x, y, w, h = cv.boundingRect(contour)
            
            # Draw the bounding box on the original frame
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Optionally, you can add a label to indicate the detected vehicle
            cv.putText(frame, 'Vehicle', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def show_frames(choice):
    back_subt=cv.createBackgroundSubtractorMOG2(history=250,varThreshold=20)
    while choice ==1:
        video_capture=capture_video("video/test3.mp4")
        if video_capture is None:
            print("Error:Could not capture video")
            break
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame or end of video.")
                break
            
            # Resize the frame to a specific size, e.g., 640x480
            resized_frame = cv.resize(frame, (900,650))
            f1=apply_background_substraction(resized_frame,back_subt)
            frame_with_vehicles = detect_vehicles(f1, resized_frame)
    
            # Display the processed frame
            cv.imshow('Vehicle Detection', frame_with_vehicles)
            cv.imshow('Captured Frame', resized_frame)
            cv.imshow("Masked Modified Omni",f1)

            if cv.waitKey(1) & 0xFF == ord('q'):
                print("Video capture stopped by user.")
                break
        # Release the capture and close the window
        video_capture.release()
        cv.destroyAllWindows()
        print("Video capture released and windows closed.")

        # Prompt user to capture video again or exit
        choice = int(input("Enter 1 to capture video again, 2 to exit: "))
    print("Program terminated.")

#main function-2
#apply background substraction
def apply_background_substraction(frame,back_subt):
    f2=back_subt.apply(frame)
    kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    f2=cv.morphologyEx(f2,cv.MORPH_OPEN,kernel)
    f2=cv.morphologyEx(f2,cv.MORPH_CLOSE,kernel)
    return f2

#Looping the program
choice=int(input("Enter choice:"))
if(choice==1):
    show_frames(choice)
elif(choice==2):
    print("Exiting the program")
    exit()
else:
    print("Wrong Choice")