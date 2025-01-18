# Human Pose Estimation using Machine Learning
A web-based application for human pose estimation using a TensorFlow-based OpenPose model. The system processes videos to detect human skeletons and outputs the results with 2D pose visualizations.

**Table of Contents**

1)Overview

2)Features

3)Technology Stack

4)Setup and Installation

5)Usage Instructions

6)Project Workflow

7)Output Examples

8)Limitations and Future Enhancements

# Overview
This project leverages a TensorFlow-based OpenPose model to perform pose estimation on videos. Users can upload videos via a web interface, and the system processes the videos to:

Identify keypoints of the human body.

Draw skeletons over the detected poses.

Save the processed videos and screenshots for review.

# Features
![image](https://github.com/user-attachments/assets/41f35602-09e7-49b1-9339-aea401087d41)

**Video Upload:** Upload videos in formats such as .mp4, .mov, .mkv, etc.

**Pose Estimation:** Detect 18 keypoints of the human body, including joints and facial landmarks.

**Real-Time Processing:** Optimized for GPU acceleration with TensorFlow for faster processing.

**Output Visualization:** View and download processed videos with skeleton overlays.

**Screenshots:** Save periodic screenshots of the processed videos for analysis.

**Privacy First:** Processes keypoints only, without storing raw video frames.

# Technology Stack
**Frameworks and Libraries**

**Frontend:** HTML, CSS, JavaScript (for the web interface).

**Backend:** Python, Flask.

**Pose Estimation:** TensorFlow, OpenPose.

**Video Processing:** OpenCV.

**Tracking:** SORT (Simple Online and Realtime Tracking).

**Hardware:** Supports both CPU and GPU processing (NVIDIA CUDA-enabled GPUs recommended for real-time performance).

# Setup and Installation

Prerequisites

Python (>=3.8)

Pip for Python package management.

CUDA and cuDNN (if using GPU acceleration).

Video files in .mp4, .mov, .mkv, or similar formats.

# Installation
**Clone this repository:**
```
git clone https://github.com/username/pose-estimation-video-processor.git
cd pose-estimation-video-processor
```
**Install dependencies:**
```
pip install -r requirements.txt
```
Place the pre-trained TensorFlow model (graph_opt.pb) in the models/ directory.

# Run the Application
**Start the Flask web server:**
```
python app.py
```
**Open the web interface in your browser:**
```
http://127.0.0.1:5000(or any local host the web server is running on but make sure the index.html file is been recognized by placing it in the templates folder in the project directory)
```

# Usage Instructions
Navigate to the web interface.

Upload a video file using the provided form.

The system will process the video, performing pose estimation frame-by-frame.

# Once completed:
Processed video will be available for download.

Screenshots of key frames will be displayed.

View them directly on the website.

# Project Workflow
**Video Upload:** Users upload a video via the web interface.

**Pose Estimation Pipeline:**

TensorFlow processes video frames to generate confidence maps and part affinity fields (PAFs).

Keypoints are extracted and connected to form a skeleton overlay.

# Output Generation:
Processed video with skeleton overlays.

Periodic screenshots of annotated frames.

**Result Display:** Outputs are stored in the outputs/ folder and displayed on the web interface for user review.

# Output Examples
**Processed Video**
https://github.com/user-attachments/assets/ceb1113e-af28-4e5b-a0f8-775ae377605a

A video with an overlay of detected skeletons on each frame.

Screenshots
![run1_frame_150](https://github.com/user-attachments/assets/74a9f992-908c-498d-b882-e0011b5288d1)
![run1_frame_100](https://github.com/user-attachments/assets/f76851c0-d69a-42bd-9e09-5a20f54f04fa)
![run1_frame_50](https://github.com/user-attachments/assets/238a7bb7-55e6-40c1-8ed4-d75ad33c6793)
![run1_frame_400](https://github.com/user-attachments/assets/641cfa52-8501-4423-bde7-a0f2199fe2a3)
![run1_frame_350](https://github.com/user-attachments/assets/39dfc040-42bb-42cd-a7b2-df128152aea5)
![run1_frame_300](https://github.com/user-attachments/assets/483fe757-62ce-4590-80f0-5abeb3fa029e)
![run1_frame_250](https://github.com/user-attachments/assets/327ba606-01eb-4d2f-b8ee-7e4fb96fea26)
![run1_frame_200](https://github.com/user-attachments/assets/b2689ba0-a3f4-4944-969f-3d01d33764d3)

Periodic images showing pose estimations at specific intervals.

# Limitations and Future Enhancements
**Current Limitations**

**Occlusion Handling:** Limited accuracy when body parts are occluded.

**Complex Environments:** May struggle in low light or cluttered backgrounds.

**2D Pose Only:** Current implementation supports only 2D pose estimation.

Future Enhancements

**3D Pose Estimation:** Extend support to 3D poses using depth cameras or stereo vision.

**Multi-Person Tracking:** Improve tracking in multi-person scenarios.

**Edge Device Deployment:** Optimize for real-time processing on edge devices like Raspberry Pi.
