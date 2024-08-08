import cv2
import os
import argparse

def create_video_from_frames(frame_folder, output_video_path, frame_rate):
    # Check if the frame folder exists
    if not os.path.exists(frame_folder):
        print(f"Error: Frame folder '{frame_folder}' does not exist.")
        return

    # Get list of all files in the frame folder
    frames = [f for f in os.listdir(frame_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    frames.sort()  # Ensure the frames are in order

    if not frames:
        print(f"Error: No frames found in the folder '{frame_folder}'.")
        return

    # Read the first frame to get the dimensions
    first_frame_path = os.path.join(frame_folder, frames[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Error: Unable to read the first frame '{first_frame_path}'.")
        return

    height, width, layers = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi, 'mp4v' for .mp4
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame in frames:
        img_path = os.path.join(frame_folder, frame)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read frame '{img_path}'. Skipping.")
            continue
        video.write(img)

    # Release the video writer
    video.release()
    print(f"Video saved to '{output_video_path}'")

def main():
    parser = argparse.ArgumentParser(description="Compose a video from frames.")
    parser.add_argument('frame_folder', type=str, help='Path to the folder containing the frames.')
    parser.add_argument('output_video_path', type=str, help='Path to the output video file.')
    parser.add_argument('frame_rate', type=int, help='Frame rate for the video.')

    args = parser.parse_args()

    create_video_from_frames(args.frame_folder, args.output_video_path, args.frame_rate)

if __name__ == "__main__":
    main()
