# Autonomous Car with YOLOv5: Ball Color Detection and Arena Ejection

This repository contains a Python-based project where an autonomous car uses the YOLOv5 object detection model to identify colored balls, navigate toward them, and push them out of an arena. The car operates autonomously, combining computer vision with robotics to perform this task efficiently.

## Project Overview

The goal of this project is to demonstrate the integration of machine learning (YOLOv5) and robotics for real-world applications:
1. **Detect**: Identify balls in an image and classify them based on color.
2. **Navigate**: Drive toward the identified ball, guided by vision-based detections.
3. **Eject**: Push the ball out of the arena to complete the task autonomously.

This project showcases the power of object detection models in robotics and autonomous systems.

## How It Works

### YOLOv5 Object Detection
- **Model**: Utilizes YOLOv5 with a custom-trained model (`balls5n.pt`) to detect and classify balls.
- **Input**: Camera feed processed through OpenCV to detect objects in real-time.
- **Output**: Bounding boxes, class labels, and confidence scores for detected objects.

### Autonomous Navigation
- The car receives object detection data (position, label, confidence) and determines the target.
- Navigation logic calculates the optimal path toward the ball.
- Once the ball is reached, the car physically pushes it out of the arena.

### Code Workflow
1. **Image Processing**: 
   - The image is read and processed to grayscale or RGB.
   - YOLOv5 performs object detection, returning predictions.
2. **Ball Selection**:
   - Filters detections to match the desired object type (e.g., "balls").
   - Selects the ball with the highest confidence score as the target.
3. **Feedback**:
   - Outputs an annotated image showing detected objects with bounding boxes and labels.

### Python Code Highlights
- YOLOv5 integration for object detection.
- OpenCV for image manipulation and visualization.
- Autonomous navigation logic to process detection data.

## Prerequisites

- Python 3.8 or later.
- Libraries:
  - `yolov5`
  - `opencv-python`
  - `numpy`
- YOLOv5 custom-trained model files:
  - `balls5n.pt` (model weights)
  - `balls5n.txt` (labels)

