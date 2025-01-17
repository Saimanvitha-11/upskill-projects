from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import subprocess

# Initialize Flask app
app = Flask(__name__, template_folder="templates")  # Specify the templates folder
CORS(app)  # Enable cross-origin requests

UPLOAD_FOLDER = "L:/STUDYYYY/Project/uploads"
OUTPUT_FOLDER = "L:/STUDYYYY/Project/outputs"
SNAPSHOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "screenshots")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOTS_FOLDER, exist_ok=True)

# Define the full path to the 'videosave.py' script
VIDEOSAVE_PATH = os.path.join(os.getcwd(), "videosave.py")

@app.route("/")
def index():
    """
    Render the main index page.
    """
    return render_template("index.html")  # Render the HTML file

@app.route("/process-video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded video
    input_video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    try:
        video_file.save(input_video_path)
    except Exception as e:
        print(f"Error saving video file: {e}")
        return jsonify({"error": "Error saving video file"}), 500

    # Run the Python script for processing the video
    try:
        subprocess.run(
            ["python", VIDEOSAVE_PATH],  # Using the full path to the videosave.py script
            check=True,
            env={
                **os.environ,
                "INPUT_VIDEO": input_video_path,
                "OUTPUT_DIR": OUTPUT_FOLDER,
            },
        )
    except subprocess.CalledProcessError as e:
        print(f"Error processing video: {e}")
        return jsonify({"error": "Error processing video"}), 500

    # Generate the output video URL and snapshots
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
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

@app.route("/outputs/screenshots/<path:filename>")
def serve_snapshots(filename):
    return send_from_directory(os.path.join(app.config["OUTPUT_FOLDER"], "screenshots"), filename)


if __name__ == "__main__":
    app.run(debug=True)
