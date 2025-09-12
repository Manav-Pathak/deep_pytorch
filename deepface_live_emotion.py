# deepface_live_emotion_opencv_pytorch.py
import cv2
import time
from deepface import DeepFace
import numpy as np

# Parameters
ANALYZE_EVERY_N_FRAMES = 15  # reduce CPU — analyze once every N frames
WEBCAM_IDX = 0              # change if you have multiple cameras

def analyze_emotion_pytorch(face_img):
    """Try PyTorch-based detectors for emotion analysis"""
    #pytorch_detectors = ['retinaface', 'mtcnn']
    
     # Resize to smaller resolution
    small_face = cv2.resize(face_img, (112, 112))  # Much faster
    
    pytorch_detectors = ['retinaface']  # Use only one detector for speed
    
    for detector in pytorch_detectors:
        try:
            # analyze expects BGR or path; enforce_detection=False avoids exception
            analysis = DeepFace.analyze(
                #face_img, 
                small_face,
                actions=['emotion'], 
                detector_backend=detector, 
                enforce_detection=False
            )
            
            # Handle case where result might be a list or dict
            if isinstance(analysis, list):
                if len(analysis) > 0:
                    analysis = analysis[0]  # Take first face if multiple detected
                else:
                    continue  # Try next detector
            
            if analysis and 'dominant_emotion' in analysis:
                dominant = analysis.get('dominant_emotion', None)
                score = None
                if dominant:
                    emotion_scores = analysis.get('emotion', {})
                    if isinstance(emotion_scores, dict) and dominant in emotion_scores:
                        score = emotion_scores[dominant]
                return (dominant, score)
                
        except Exception as e:
            # Try next detector if this one fails
            print(f"Error with {detector}: {e}")
            continue
    
    return None  # All detectors failed

def main():
    cap = cv2.VideoCapture(WEBCAM_IDX)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Load OpenCV's Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    last_result = None  # store last analyze result for display
    start_time = time.time()

    print("Starting live emotion detection with OpenCV + PyTorch backends...")
    print("Press 'q' or ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using OpenCV
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Process only the first face
            x, y, bw, bh = faces[0]
            x2 = x + bw
            y2 = y + bh

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            # Every N frames, run DeepFace analyze on cropped face
            if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                # expand bbox a little
                pad = int(0.15 * max(bw, bh))
                xa = max(0, x - pad)
                ya = max(0, y - pad)
                xb = min(w, x2 + pad)
                yb = min(h, y2 + pad)

                face_img = frame[ya:yb, xa:xb]
                if face_img.size != 0:
                    result = analyze_emotion_pytorch(face_img)
                    if result:
                        last_result = result
                    # If result is None, keep the last_result for display
                else:
                    last_result = None

            # Draw last_result text in top-left of box
            if last_result and last_result[0] is not None:
                dominant, score = last_result
                # Format score; if None, display '?'
                score_text = f"{score:.2f}%" if (score is not None) else "?"
                text = f"{dominant} {score_text}"
                # Put text with background for readability
                (tx, ty), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                # background rectangle
                cv2.rectangle(frame, (x, y - ty - baseline - 6), (x + tx + 6, y), (0, 255, 0), -1)
                cv2.putText(frame, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        else:
            # Optionally show "No face"
            cv2.putText(frame, "No face", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # Show FPS counter
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("DeepFace Live Emotion (OpenCV + PyTorch)", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()