# 📸 Day 14: Real-Time Object Detection with YOLOv5

This is the fourteenth project of my #30DaysOfAI challenge. This project leverages a state-of-the-art deep learning model, **YOLOv5**, to perform real-time object detection on images. The app is multilingual (TR/EN).

### ✨ Key Concepts
* **Object Detection:** A computer vision task that deals with detecting instances of semantic objects in digital images.
* **YOLO (You Only Look Once):** A family of one-stage object detection models that are fast and accurate.
* **Pre-trained Models & PyTorch Hub:** This project utilizes a powerful, pre-trained YOLOv5s model directly from PyTorch Hub, demonstrating the critical engineering skill of leveraging existing state-of-the-art models.

### 💻 Tech Stack
- Python, Streamlit, PyTorch, YOLOv5, OpenCV, Ultralytics

### 🚀 How to Run
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *This command reads the `requirements.txt` file and automatically installs all the necessary libraries (like `streamlit`, `torch`, `ultralytics`, etc.) that the project needs to run. This ensures the code works correctly in any environment.*

2.  **Run the app:** (No training step needed!)
    ```bash
    streamlit run app.py
    ```