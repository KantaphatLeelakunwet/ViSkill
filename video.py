import cv2
import os
import sys
from natsort import natsorted


def create_video_from_images(image_folder, video_name, fps=30):
    # Get all image files from the folder
    images = [img for img in os.listdir(
        image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort images by filename
    images = natsorted(images)

    # Check if there are images in the folder
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Read and write each image to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # Write the frame to the video

    # Release the video writer
    video.release()
    print(f"Video '{video_name}' created successfully.")


if __name__ == "__main__":
    fps = 10  # Frames per second
    image_folder = f'./saved_eval_pic/'
    video_name = image_folder + f"{sys.argv[1]}.mp4" # 'output_video.mp4'  # Desired output video name
    create_video_from_images(image_folder, video_name, fps)
