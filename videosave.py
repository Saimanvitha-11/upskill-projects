import cv2
import os
import numpy as np

# Define body parts and pose pairs
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
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            out_blob = net.forward()
            out_blob = out_blob[:, :19, :, :]

            points = []
            for i, part in enumerate(BODY_PARTS.keys()):
                if part == "Background":
                    continue
                heatMap = out_blob[0, i, :, :]
                _, conf, _, point = cv2.minMaxLoc(heatMap)
                x = int((frame_width * point[0]) / out_blob.shape[3])
                y = int((frame_height * point[1]) / out_blob.shape[2])
                points.append((x, y) if conf > 0.2 else None)

                if conf > 0.2:
                    cv2.putText(frame, part, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[part], 2, cv2.LINE_AA)

            for pair in POSE_PAIRS:
                partFrom, partTo = pair
                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]

                if points[idFrom] and points[idTo]:
                    cv2.line(frame, points[idFrom], points[idTo], COLORS[partFrom], 3)
                    cv2.ellipse(frame, points[idFrom], (5, 5), 0, 0, 360, COLORS[partFrom], cv2.FILLED)
                    cv2.ellipse(frame, points[idTo], (5, 5), 0, 0, 360, COLORS[partTo], cv2.FILLED)

            out.write(frame)

            # Save a screenshot every 2 seconds
            if frame_count % int(fps * 1) == 0:
                screenshot_path = os.path.join(screenshot_dir, f"{os.path.splitext(video_file)[0]}_frame_{frame_count}.png")
                cv2.imwrite(screenshot_path, frame)

            # Display the live processed video
            cv2.imshow('Processed Video', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()

    cv2.destroyAllWindows()
    print(f"Processing completed. Videos saved in: {output_dir}")
    print(f"Screenshots saved in: {screenshot_dir}")

# Define directories
input_dir = "L:/STUDYYYY/Project/uploads"
output_dir = "L:/STUDYYYY/Project/outputs"

process_videos(input_dir, output_dir)
