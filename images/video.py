import numpy as np
import cv2
import sys

# Assuming you have a NumPy array of shape (100, 480, 640, 3)
# For demonstration, let's create a random array
num_frames = 100
height = 480
width = 640
channels = 3

# Create a random NumPy array (replace this with your actual data)
input_filename = sys.argv[1]
video_array = np.load(input_filename)

# Define the codec and create a VideoWriter object for MP4
output_filename = f'{input_filename[:-4]}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
fps = 10  # Frames per second

# Create a VideoWriter object
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

# Write each frame to the video
for i in range(num_frames):
    frame = video_array[i]
    out.write(frame)

# Release the VideoWriter object
out.release()

print(f"Video saved as {output_filename}")