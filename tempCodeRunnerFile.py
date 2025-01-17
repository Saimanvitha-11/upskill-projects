from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import subprocess
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "L:/STUDYYYY/Project/uploads"
OUTPUT_FOLDER = "L:/STUDYYYY/Project/outputs"
SNAPSHOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "screenshots")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOTS_FOLDER, exist_ok=True)

@app.route("/process-video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded video
    input_video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(input_video_path)

    # Run the Python script for processing
    try:
        subprocess.run(
            ["python", "videosave.py"],
            check=True,
            env={
                **os.environ,
                "INPUT_VIDEO": input_video_path,
                "OUTPUT_DIR": OUTPUT_FOLDER,
            },
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Error processing video"}), 500

    # Generate output URLs
    processed_video_path = os.path.join(OUTPUT_FOLDER, f"processed_{video_file.filename}")
    snapshots = [
        f"/outputs/screenshots/{filename}"
        for filename in os.listdir(SNAPSHOTS_FOLDER)
        if filename.startswith(os.path.splitext(video_file.filename)[0])
    ]

    return jsonify({
        "video_url": f"/outputs/processed_{video_file.filename}",
        "snapshots": snapshots
    })

@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/outputs/screenshots/<path:filename>")
def serve_snapshots(filename):
    return send_from_directory(SNAPSHOTS_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
