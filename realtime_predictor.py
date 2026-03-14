import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Tuple, Dict, Any

class PCOSDetector:
    """
    A class to handle real-time facial analysis for PCOS detection using a webcam feed.

    This class encapsulates the model loading, face detection, image preprocessing,
    and prediction logic.
    
    Attributes:
        MODEL_PATH (str): Path to the trained Keras model file.
        CASCADE_PATH (str): Path to the OpenCV Haar Cascade XML file for face detection.
        CLASS_LABELS (List[str]): The labels for the prediction classes.
                                   **Must match the training order.**
        BOX_COLORS (Dict[str, Tuple[int, int, int]]): Colors for drawing bounding boxes.
    """
    MODEL_PATH: str = 'model.h5'
    CASCADE_PATH: str = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    CLASS_LABELS: List[str] = ['Healthy', 'PCOS Positive']
    BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
        'Healthy': (0, 255, 0),       # Green
        'PCOS Positive': (0, 0, 255)  # Red
    }

    def __init__(self):
        """Initializes the detector by loading the model and face cascade."""
        self.model = self._load_keras_model()
        self.face_cascade = self._load_cascade_classifier()
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise IOError("❌ Error: Could not open video stream.")

    def _load_keras_model(self) -> tf.keras.Model:
        """Loads the trained Keras model from the specified path."""
        try:
            model = load_model(self.MODEL_PATH)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            raise IOError(f"❌ Error loading model: {e}")

    def _load_cascade_classifier(self) -> cv2.CascadeClassifier:
        """Loads the Haar Cascade classifier for face detection."""
        face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        if face_cascade.empty():
            raise IOError("❌ Error loading face cascade. Check the OpenCV installation or file path.")
        return face_cascade

    def _process_predictions(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]], predictions: np.ndarray) -> None:
        """Annotates the frame with bounding boxes and prediction labels."""
        for (x, y, w, h), prediction in zip(faces, predictions):
            # --- CORRECT PREDICTION LOGIC FOR SOFTMAX ---
            # Use argmax to find the index of the highest probability
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction)
            label = self.CLASS_LABELS[predicted_index]
            
            # Prepare text and color for display
            label_text = f"{label} ({confidence:.2f})"
            color = self.BOX_COLORS[label]

            # Draw rectangle and text on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y + h + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)
            
    def run(self) -> None:
        """Starts the real-time detection loop."""
        print("🚀 Starting video stream... Press 'q' to quit.")
        
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                print("⚠️ Warning: Could not read frame. Exiting...")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale frame
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )

            if len(faces) > 0:
                # Prepare a batch of face images for prediction
                face_batch = []
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y+h, x:x+w]
                    # Preprocess for the model
                    resized_face = cv2.resize(face_roi, (224, 224))
                    normalized_face = resized_face / 255.0
                    face_batch.append(normalized_face)
                
                # Predict on the entire batch at once for efficiency
                predictions = self.model.predict(np.array(face_batch))
                
                # Annotate the frame with the results
                self._process_predictions(frame, faces, predictions)

            cv2.imshow('PCOS Facial Recognition (Educational Demo)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # --- Cleanup ---
        self.video_capture.release()
        cv2.destroyAllWindows()
        print("👋 Video stream stopped.")

if __name__ == "__main__":
    try:
        detector = PCOSDetector()
        detector.run()
    except IOError as e:
        print(e)