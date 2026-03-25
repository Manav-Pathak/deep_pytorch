# SpeakSmart - Speech and Video Analysis System

SpeakSmart is a multimodal communication-analysis project that evaluates speaking performance from uploaded audio/video. It combines speech speed, audio sentiment, facial emotion analysis, eye-contact scoring, and posture assessment to generate practical presentation feedback.

## Key Features

- Streamlit app for uploading video or audio and viewing analysis results
- Parallel audio and video processing for faster end-to-end analysis
- Speech speed estimation (WPM) and pace categorization (Slow/Normal/Fast)
- Audio sentiment prediction using pre-trained ML models
- Facial emotion analysis from video using OpenCV + DeepFace
- Eye-contact analysis using MediaPipe FaceMesh landmarks
- Posture analysis using MediaPipe Pose + trained SVM classifier
- Processed output videos with visual overlays and summary metrics

## Tech Stack

### Core

- Python
- Streamlit

### Computer Vision

- OpenCV
- DeepFace
- MediaPipe

### Audio and ML

- Librosa
- SpeechRecognition
- Scikit-learn
- XGBoost
- Joblib

## Basic Project Structure

```text
deep_pytorch/
|-- app.py                       # Main Streamlit app (parallel audio + video pipeline)
|-- video_processor.py           # Face detection + DeepFace emotion processing
|-- Eye_Contact/
|   |-- eyecontact.py            # Eye-contact scoring with MediaPipe FaceMesh
|-- Posture/
|   |-- posture_utils.py         # Posture inference pipeline
|   |-- train.py                 # SVM training script for posture classifier
|   |-- extract_keypoints.py     # Pose keypoint dataset extraction utility
|-- Sentiment Analysis/          # Sentiment training/inference utilities
|-- Speech Speed/                # Speech speed related utilities
```

## Sample Outputs

<p align="center">
	<img width="400" height="300" alt="Output 1" src="https://github.com/user-attachments/assets/6518b59a-7f26-4b97-8437-aba8d178f166" />
</p>

<p align="center">
	<img width="400" height="300" alt="Output 2" src="https://github.com/user-attachments/assets/358d330f-5d0b-4e7e-8922-eacfa7ecd2af" />
</p>

<p align="center">
  <img width="600" height="500" alt="Output 3" src="https://github.com/user-attachments/assets/c84540de-2c2c-4796-b64f-317e9f8cfd56"  />
</p>

## Run Locally

### 1) Clone and install dependencies

```bash
git clone https://github.com/Manav-Pathak/deep_pytorch.git
cd deep_pytorch
pip install -r requirements.txt
```

### 2) Install FFmpeg (required for video-to-audio extraction)

- Windows: install FFmpeg and ensure `ffmpeg` is available in PATH
- Verify:

```bash
ffmpeg -version
```

### 3) Ensure model files are present

Required model artifacts:

- `xgboost_model.pkl`
- `scaler.pkl`
- `label_encoder.pkl`
- `Posture/posture_svm.pkl`
- `Posture/scaler.pkl`
- `Posture/label_encoder.pkl`

### 4) Start the app

```bash
streamlit run app.py
```

Open the app at the local URL shown in terminal (typically `http://localhost:8501`).

## Notes

- First DeepFace run may download model weights and take longer.
- Video emotion analysis is sampled every N frames for performance.
- Best results are obtained with clear face visibility, stable lighting, and audible speech.
