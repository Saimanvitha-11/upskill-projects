import cv2
import os
import numpy as np
from sort.sort import Sort  # Sorting library for tracking multiple people
from skimage import io

# Define body parts and pose pairs (ensure this is defined before use)
BODY_PARTS = { 
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 
}

POSE_PAIRS = [ 
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] 
]

COLORS = {part: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for part in BODY_PARTS}

# Load the pre-trained model
try:
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize the tracker (SORT)
tracker = Sort()

def process_videos(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    screenshot_dir = os.path.join(output_dir, "screenshots")
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    for video_file in video_files:
        input_video_path = os.path.join(input_dir, video_file)
        output_video_path = os.path.join(output_dir, f"processed_{video_file}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {input_video_path}")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    raise ValueError("Error reading frame from video")

                frame_count += 1

                # Display processing information on the frame
                progress_text = f"Processing frame {frame_count}/{total_frames}"
                cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
                out_blob = net.forward()
                out_blob = out_blob[:, :19, :, :]

                points = []
                detections = []
                for i, part in enumerate(BODY_PARTS.keys()):
                    if part == "Background":
                        continue
                    heatMap = out_blob[0, i, :, :]
                    _, conf, _, point = cv2.minMaxLoc(heatMap)
                    x = int((frame_width * point[0]) / out_blob.shape[3])
                    y = int((frame_height * point[1]) / out_blob.shape[2])
                    points.append((x, y) if conf > 0.3 else None)

                    if conf > 0.3:
                        # Confidence score display
                        cv2.putText(frame, f"{part}: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[part], 2, cv2.LINE_AA)
                        # Detection for SORT (4 values for bbox)
                        detections.append([x, y, 10, 10])  # 10x10 as a placeholder size for tracking

                # Update trackers with new detections
                trackers = tracker.update(np.array(detections))
                for tracker_info in trackers:
                    tracker_id = int(tracker_info[4])  # Get unique ID from SORT
                    x1, y1, x2, y2 = map(int, tracker_info[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {tracker_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw skeleton layout
                for pair in POSE_PAIRS:
                    partA, partB = pair
                    if points[BODY_PARTS[partA]] and points[BODY_PARTS[partB]]:
                        cv2.line(frame, points[BODY_PARTS[partA]], points[BODY_PARTS[partB]], (0, 255, 255), 2)
                        cv2.circle(frame, points[BODY_PARTS[partA]], 4, (0, 0, 255), -1)
                        cv2.circle(frame, points[BODY_PARTS[partB]], 4, (0, 0, 255), -1)

                # Save the processed frame to output video
                out.write(frame)

                # Save screenshots every 2 seconds
                if frame_count % int(fps * 2) == 0:
                    screenshot_path = os.path.join(screenshot_dir, f"{os.path.splitext(video_file)[0]}_frame_{frame_count}.png")
                    cv2.imwrite(screenshot_path, frame)

                # Display the live processed video
                display_height = 720
                aspect_ratio = frame_width / frame_height
                display_width = int(display_height * aspect_ratio)
                resized_frame = cv2.resize(frame, (display_width, display_height))
                cv2.imshow('Processed Video', resized_frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                print(f"Error in frame processing: {e}")
                break

        cap.release()
        out.release()

    cv2.destroyAllWindows()
    print(f"Processing completed. Videos saved in: {output_dir}")
    print(f"Screenshots saved in: {screenshot_dir}")

# Define directories
input_dir = "L:/STUDYYYY/pose estimation/uploads"
output_dir = "L:/STUDYYYY/pose estimation/outputs"

process_videos(input_dir, output_dir)
