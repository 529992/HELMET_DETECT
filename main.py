import cv2
import tkinter as tk
from PIL import Image, ImageTk

# Load the pre-trained Haar Cascade classifier for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
head_cascade = cv2.CascadeClassifier(cascade_path)

# Function to capture and display video frames
def show_frame():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect heads (faces) in the frame
        heads = head_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around the detected heads
        for (x, y, w, h) in heads:
            # Estimate the area above the face (for hat detection)
            hat_region = gray[max(0, y - int(h * 0.25)):y, x:x + w]
            
            # Check for the presence of a hat by analyzing the region above the face
            # We'll use a simple brightness-based threshold here (you can improve this with better detection logic)
            hat_present = cv2.mean(hat_region)[0] < 90  # If the region is darker, assume it's a hat
            
            # Set color: Blue for a hat, Red for no hat
            if hat_present:
                color = (255, 0, 0)  # Blue (BGR format)
            else:
                color = (0, 0, 255)  # Red (BGR format)
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Convert the frame to RGB (PIL format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a PIL image and then to an ImageTk object
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new frame
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    
    # Schedule the function to be called again after 10 ms
    lbl_video.after(10, show_frame)

# Initialize tkinter
root = tk.Tk()
root.title("Hat Detection in Real-Time")

# Create a Label widget to display the video
lbl_video = tk.Label(root)
lbl_video.pack()

# Start video capture
cap = cv2.VideoCapture(0)

# Call the show_frame function to start the video capture loop
show_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the video capture object when the window is closed
cap.release()
cv2.destroyAllWindows()
