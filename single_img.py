# deepface_test_image_pytorch_only.py
import sys
import cv2
import numpy as np

try:
    from deepface import DeepFace
except Exception as e:
    print("Failed to import DeepFace:", e)
    sys.exit(1)

print("DeepFace imported OK")

# Put path to your test image here:
IMG_PATH = "anxious_woman.jpg"  # change to your file

# Quick check: load image
img = cv2.imread(IMG_PATH)
if img is None:
    print(f"Could not read image '{IMG_PATH}'. Make sure file exists and path is correct.")
    sys.exit(1)

print(f"Image loaded successfully. Shape: {img.shape}")

# Try different PyTorch-based detectors
pytorch_detectors = ['retinaface', 'mtcnn']
success = False  # Add success flag

for detector in pytorch_detectors:
    try:
        print(f"\nTrying PyTorch-based detector: {detector}")
        # Set enforce_detection=False to handle cases where no face is detected
        result = DeepFace.analyze(
            img_path=IMG_PATH, 
            actions=['emotion'], 
            detector_backend=detector,
            enforce_detection=False  # This prevents the error when no face is found
        )
        
        # Handle case where result might be a list or dict
        if isinstance(result, list):
            if len(result) > 0:
                result = result[0]  # Take first face if multiple detected
            else:
                print(f"No faces detected with {detector}")
                continue
        
        if result and 'dominant_emotion' in result:
            print("Analysis result:")
            print(result)
            dominant = result.get('dominant_emotion', None)
            if dominant:
                emotion_scores = result.get('emotion', {})
                if isinstance(emotion_scores, dict) and dominant in emotion_scores:
                    score = emotion_scores[dominant]
                    print(f"Dominant: {dominant} ({score:.2f}%)")
                else:
                    print(f"Dominant: {dominant}")
            
            # Print all emotion scores
            if 'emotion' in result:
                print("\nAll emotion scores:")
                for emotion, score in result['emotion'].items():
                    print(f"  {emotion}: {score:.2f}%")
            
            print(f"\nSuccess with PyTorch detector: {detector}")
            success = True  # Set success flag
            break
        else:
            print(f"No valid result from {detector}")
            
    except Exception as e:
        print(f"Error with {detector}: {e}")
        continue
else:
    print("\nAll PyTorch detectors failed. Possible solutions:")
    print("1. Check if the image actually contains a clear face")
    print("2. Try preprocessing the image (resize, enhance contrast)")
    print("3. Use a different image")
    print("4. Make sure PyTorch and required dependencies are installed")

print("\nDebugging info:")
print(f"Image path: {IMG_PATH}")
print(f"Image shape: {img.shape if img is not None else 'None'}")

