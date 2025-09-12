import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import speech_recognition as sr_module
import os
import threading
import time
import subprocess
from pathlib import Path
import cv2
from deepface import DeepFace
from video_processor import process_video, get_emotion_group

st.set_page_config(page_title="SpeakSmart", layout="centered")
st.title("🎤 Speech & Video Analysis")

# Create extracted folder if it doesn't exist
EXTRACTED_FOLDER = "extracted"
os.makedirs(EXTRACTED_FOLDER, exist_ok=True)

# Global variables for parallel processing
audio_results = {}
video_results = {}
processing_status = {"audio": False, "video": False}

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video and save as .wav file"""
    try:
        # Use ffmpeg to extract audio
        command = [
            'ffmpeg', '-i', video_path, 
            '-vn',  # no video
            '-acodec', 'pcm_s16le',  # audio codec
            '-ar', '16000',  # sample rate
            '-ac', '1',  # mono channel
            '-y',  # overwrite output file
            output_audio_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            st.error(f"Audio extraction failed: {result.stderr}")
            return False
    except Exception as e:
        # Fallback to librosa if ffmpeg fails
        try:
            y, sr = librosa.load(video_path, sr=16000)
            librosa.output.write_wav(output_audio_path, y, sr)
            return True
        except Exception as e2:
            st.error(f"Audio extraction failed with both methods: {str(e2)}")
            return False

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

def extract_wpm(file_path):
    y, sample_rate = librosa.load(file_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sample_rate)

    recognizer = sr_module.Recognizer()
    with sr_module.AudioFile(file_path) as source:
        try:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            word_count = len(text.split())
            wpm = (word_count / duration) * 60
        except:
            wpm = 0
    return wpm

def categorize_wpm(wpm):
    if wpm < 160:
        return "Slow"
    elif 160 <= wpm <= 195:
        return "Normal"
    else:
        return "Fast"

def audio_analysis_thread(audio_path, video_name):
    """Thread function for audio analysis"""
    global audio_results, processing_status
    
    try:
        processing_status["audio"] = True
        
        # Extract WPM
        wpm = extract_wpm(audio_path)
        category = categorize_wpm(wpm)
        
        # Extract features and predict sentiment
        features = extract_features(audio_path)
        features_scaled = scaler.transform(features.reshape(1, -1))
        pred = xgb.predict(features_scaled)
        pred_sentiment = le.inverse_transform(pred)[0]
        
        audio_results[video_name] = {
            "wpm": round(wpm),
            "category": category,
            "sentiment": pred_sentiment,
            "status": "completed"
        }
        
    except Exception as e:
        audio_results[video_name] = {
            "status": "error",
            "error": str(e)
        }
    finally:
        processing_status["audio"] = False

def video_analysis_thread(video_path, video_name):
    """Thread function for video analysis"""
    global video_results, processing_status
    
    try:
        processing_status["video"] = True
        video_results[video_name] = {"status": "processing", "progress": 0}
        
        # Create unique output path for this video
        output_path = f"deepface_processed/{video_name}_emotion_analyzed.mp4"
        
        # Process video (this will take longer)
        success, emotion_analysis = process_video(video_path, output_path)
        
        if success:
            video_results[video_name] = {
                "status": "completed",
                "success": True,
                "output_path": output_path,
                "emotion_analysis": emotion_analysis
            }
        else:
            video_results[video_name] = {
                "status": "error",
                "error": "Video processing failed"
            }
            
    except Exception as e:
        video_results[video_name] = {
            "status": "error", 
            "error": str(e)
        }
    finally:
        processing_status["video"] = False

uploaded_file = st.file_uploader("Upload your video file (.mp4, .avi, .mov) or audio file (.wav, .mp3)", 
                                type=["mp4", "avi", "mov", "wav", "mp3"])

xgb = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    is_video = file_extension in ['mp4', 'avi', 'mov']
    is_audio = file_extension in ['wav', 'mp3']
    
    if is_video:
        # Save uploaded video with unique name
        video_name = uploaded_file.name.split('.')[0]
        video_path = f"temp_{video_name}.{file_extension}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.video(video_path)
        
        # Extract audio from video
        video_name = uploaded_file.name.split('.')[0]
        audio_path = os.path.join(EXTRACTED_FOLDER, f"{video_name}_extracted.wav")
        
        st.subheader("🎵 Extracting Audio...")
        audio_extraction_success = extract_audio_from_video(video_path, audio_path)
        
        if audio_extraction_success:
            st.success("✅ Audio extracted successfully!")
            st.audio(audio_path)
            
            # Start parallel processing
            st.subheader("🔄 Starting Parallel Analysis...")
            
            # Start both threads
            audio_thread = threading.Thread(target=audio_analysis_thread, args=(audio_path, video_name))
            video_thread = threading.Thread(target=video_analysis_thread, args=(video_path, video_name))
            
            audio_thread.start()
            video_thread.start()
            
            # Create placeholders for real-time updates
            audio_placeholder = st.empty()
            video_placeholder = st.empty()
            results_placeholder = st.empty()
            
            # Monitor progress
            while audio_thread.is_alive() or video_thread.is_alive():
                with audio_placeholder.container():
                    if video_name in audio_results:
                        if audio_results[video_name]["status"] == "completed":
                            st.success("🎤 Audio Analysis Complete!")
                        elif audio_results[video_name]["status"] == "error":
                            st.error(f"❌ Audio Analysis Error: {audio_results[video_name]['error']}")
                    else:
                        st.info("🎤 Audio Analysis in progress...")
                
                with video_placeholder.container():
                    if video_name in video_results:
                        if video_results[video_name]["status"] == "completed":
                            st.success("🎬 Video Analysis Complete!")
                        elif video_results[video_name]["status"] == "error":
                            st.error(f"❌ Video Analysis Error: {video_results[video_name]['error']}")
                        elif video_results[video_name]["status"] == "processing":
                            st.info("🎬 Video Analysis in progress...")
                    else:
                        st.info("🎬 Video Analysis starting...")
                
                time.sleep(1)  # Update every second
            
            # Wait for threads to complete
            audio_thread.join()
            video_thread.join()
            
        else:
            st.error("❌ Failed to extract audio from video")
            
    elif is_audio:
        # Handle direct audio upload (existing functionality)
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        
        st.audio("temp.wav")
        audio_path = "temp.wav"
        video_name = uploaded_file.name.split('.')[0]
        
        # Start audio analysis thread
        audio_thread = threading.Thread(target=audio_analysis_thread, args=(audio_path, video_name))
        audio_thread.start()
        
        audio_placeholder = st.empty()
        
        while audio_thread.is_alive():
            with audio_placeholder.container():
                st.info("🎤 Audio Analysis in progress...")
            time.sleep(1)
        
        audio_thread.join()

# Display results section
if uploaded_file is not None:
    # Get file type info
    file_extension = uploaded_file.name.split('.')[-1].lower()
    is_video = file_extension in ['mp4', 'avi', 'mov']
    is_audio = file_extension in ['wav', 'mp3']
    
    # Display Results
    st.subheader("📊 Analysis Results")
    
    # Get the video name for results lookup
    video_name = uploaded_file.name.split('.')[0]
    
    # Audio Results
    if video_name in audio_results:
        audio_result = audio_results[video_name]
        if audio_result["status"] == "completed":
            st.success("🎤 **Audio Analysis Results:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Speech Speed (WPM)", audio_result["wpm"])
                st.metric("Speed Category", audio_result["category"])
            
            with col2:
                st.metric("Audio Sentiment", audio_result["sentiment"].upper())
                
                # Show feedback based on sentiment
                feedback_messages = {
                    "positive": "Great job! You sound confident and engaging. 🚀",
                    "negative": "Try to stay calm and positive, it will improve your delivery. 🌱",
                    "neutral": "Your tone is balanced and professional. 👍"
                }
                
                if audio_result["sentiment"] in feedback_messages:
                    st.info(feedback_messages[audio_result["sentiment"]])
        
        elif audio_result["status"] == "error":
            st.error(f"Audio Analysis Error: {audio_result['error']}")
    
    # Video Results (only for video uploads)
    if is_video and video_name in video_results:
        video_result = video_results[video_name]
        if video_result["status"] == "completed" and video_result["success"]:
            st.success("🎬 **Video Analysis Complete!**")
            
            # Display facial emotion analysis results
            if "emotion_analysis" in video_result and video_result["emotion_analysis"]:
                emotion_data = video_result["emotion_analysis"]
                
                st.subheader("😊 **Facial Emotion Analysis:**")
                
                if emotion_data.get("total_detections", 0) > 0:
                    # Create columns for emotion breakdown
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if "positive" in emotion_data:
                            st.metric("Positive Emotions", 
                                    f"{emotion_data['positive']['percentage']}%",
                                    f"{emotion_data['positive']['count']} detections")
                    
                    with col2:
                        if "negative" in emotion_data:
                            st.metric("Negative Emotions", 
                                    f"{emotion_data['negative']['percentage']}%",
                                    f"{emotion_data['negative']['count']} detections")
                    
                    with col3:
                        if "neutral" in emotion_data:
                            st.metric("Neutral Emotions", 
                                    f"{emotion_data['neutral']['percentage']}%",
                                    f"{emotion_data['neutral']['count']} detections")
                    
                    # Show dominant emotion
                    if emotion_data.get("dominant_emotion"):
                        dominant = emotion_data["dominant_emotion"]
                        dominant_pct = emotion_data.get("dominant_percentage", 0)
                        
                        # Color code based on dominant emotion
                        if dominant == "positive":
                            st.success(f"🌟 **Dominant Facial Emotion: {dominant.upper()}** ({dominant_pct}%)")
                        elif dominant == "negative":
                            st.error(f"😔 **Dominant Facial Emotion: {dominant.upper()}** ({dominant_pct}%)")
                        else:
                            st.info(f"😐 **Dominant Facial Emotion: {dominant.upper()}** ({dominant_pct}%)")
                    
                    st.info(f"📊 Total emotion detections: {emotion_data['total_detections']}")
                else:
                    st.warning("⚠️ No facial emotions detected in the video")
            
            # Show processed video
            if os.path.exists(video_result["output_path"]):
                st.video(video_result["output_path"])
                st.download_button(
                    label="📥 Download Processed Video",
                    data=open(video_result["output_path"], "rb").read(),
                    file_name=f"{video_name}_emotion_analyzed.mp4",
                    mime="video/mp4"
                )
            
        elif video_result["status"] == "error":
            st.error(f"Video Analysis Error: {video_result['error']}")
    
    # Combined Summary
    if video_name in audio_results and audio_results[video_name]["status"] == "completed":
        if is_video and video_name in video_results and video_results[video_name]["status"] == "completed":
            st.subheader("🎯 Combined Analysis Summary")
            st.write("**Overall Assessment:**")
            
            audio_sentiment = audio_results[video_name]["sentiment"]
            speed_category = audio_results[video_name]["category"]
            
            # Get facial emotion data if available
            facial_emotion = None
            if "emotion_analysis" in video_results[video_name] and video_results[video_name]["emotion_analysis"]:
                emotion_data = video_results[video_name]["emotion_analysis"]
                if emotion_data.get("total_detections", 0) > 0:
                    facial_emotion = emotion_data.get("dominant_emotion")
            
            # Create comprehensive assessment
            assessment_factors = []
            positive_factors = 0
            total_factors = 0
            
            # Audio sentiment assessment
            if audio_sentiment == "positive":
                assessment_factors.append("✅ Positive audio sentiment")
                positive_factors += 1
            elif audio_sentiment == "negative":
                assessment_factors.append("❌ Negative audio sentiment")
            else:
                assessment_factors.append("⚪ Neutral audio sentiment")
                positive_factors += 0.5
            total_factors += 1
            
            # Speech speed assessment
            if speed_category == "Normal":
                assessment_factors.append("✅ Good speech pacing")
                positive_factors += 1
            elif speed_category == "Slow":
                assessment_factors.append("🐌 Speech is too slow")
            else:
                assessment_factors.append("🏃 Speech is too fast")
            total_factors += 1
            
            # Facial emotion assessment
            if facial_emotion:
                if facial_emotion == "positive":
                    assessment_factors.append("✅ Positive facial expressions")
                    positive_factors += 1
                elif facial_emotion == "negative":
                    assessment_factors.append("❌ Negative facial expressions")
                else:
                    assessment_factors.append("⚪ Neutral facial expressions")
                    positive_factors += 0.5
                total_factors += 1
            else:
                assessment_factors.append("⚠️ No facial emotions detected")
            
            # Display assessment factors
            for factor in assessment_factors:
                st.write(f"• {factor}")
            
            # Overall score and recommendation
            if total_factors > 0:
                overall_score = (positive_factors / total_factors) * 100
                
                st.write("---")
                if overall_score >= 80:
                    st.success(f"🌟 **Excellent Performance!** (Score: {overall_score:.0f}%)")
                    st.write("Your communication is very effective across all dimensions.")
                elif overall_score >= 60:
                    st.info(f"👍 **Good Performance** (Score: {overall_score:.0f}%)")
                    st.write("Strong communication with room for minor improvements.")
                elif overall_score >= 40:
                    st.warning(f"⚡ **Needs Improvement** (Score: {overall_score:.0f}%)")
                    st.write("Several areas could be enhanced for better communication.")
                else:
                    st.error(f"🎯 **Significant Improvement Needed** (Score: {overall_score:.0f}%)")
                    st.write("Focus on improving multiple aspects of your communication.")
                
                # Specific recommendations
                recommendations = []
                if audio_sentiment == "negative":
                    recommendations.append("🎭 Work on maintaining a more positive tone")
                if speed_category != "Normal":
                    if speed_category == "Slow":
                        recommendations.append("⚡ Try speaking slightly faster for better engagement")
                    else:
                        recommendations.append("🐌 Slow down your speech for better clarity")
                if facial_emotion == "negative":
                    recommendations.append("😊 Practice more positive facial expressions")
                elif not facial_emotion:
                    recommendations.append("📹 Ensure good lighting and camera positioning for emotion detection")
                
                if recommendations:
                    st.write("**💡 Recommendations:**")
                    for rec in recommendations:
                        st.write(f"• {rec}")
