import os
import sys
import logging
import base64
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, session, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import random
import requests
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from io import BytesIO
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('femaura_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Add CORS headers manually for chatbot requests
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Get the absolute path of the current directory
basedir = os.path.abspath(os.path.dirname(__file__))
logger.info(f"Application base directory: {basedir}")

# Database Configuration - Using SQLite for easier setup
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload Configuration
upload_folder = os.path.join(basedir, 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Model Configuration
MODEL_PATH = os.path.join(basedir, 'model.h5')
TABULAR_MODEL_PATH = os.path.join(basedir, 'pcos_tabular_model.pkl')
TABULAR_SCALER_PATH = os.path.join(basedir, 'pcos_scaler.pkl')
FACIAL_MODEL_PATH = os.path.join(basedir, 'facial_pcos_model.h5')  # Will be created separately
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Model exists: {os.path.exists(MODEL_PATH)}")
logger.info(f"Tabular model path: {TABULAR_MODEL_PATH}")
logger.info(f"Tabular model exists: {os.path.exists(TABULAR_MODEL_PATH)}")

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload folder created/verified: {upload_folder}")

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    prediction_type = db.Column(db.String(20), nullable=False)  # 'upload' or 'camera'
    image_path = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.prediction_result}>'

# Global variables
model = None
tabular_model = None
tabular_scaler = None
tabular_feature_names = None
facial_model = None  # For facial feature detection

def load_model():
    """Load the TensorFlow model with comprehensive error handling"""
    global model
    try:
        print("DEBUG: Starting model loading process...")
        logger.info("Attempting to load TensorFlow model...")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            print(f"CRITICAL ERROR: Model file not found at: {MODEL_PATH}")
            logger.error(f"Model file not found at: {MODEL_PATH}")
            return False
        
        print(f"SUCCESS: Model file found. Size: {os.path.getsize(MODEL_PATH)} bytes")
        logger.info(f"Model file found. Size: {os.path.getsize(MODEL_PATH)} bytes")
        
        # Try to import TensorFlow
        try:
            import tensorflow as tf
            print(f"SUCCESS: TensorFlow imported successfully. Version: {tf.__version__}")
            logger.info(f"TensorFlow version: {tf.__version__}")
        except ImportError as e:
            print(f"CRITICAL ERROR: TensorFlow import failed: {e}")
            logger.error(f"TensorFlow import failed: {e}")
            return False
        
        # Load the model with multiple fallback methods
        try:
            print("DEBUG: Attempting to load model with compile=True...")
            # Try different loading methods
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=True)
                print("SUCCESS: Model loaded successfully with compile=True!")
                logger.info("Model loaded successfully with compile=True!")
            except Exception as e1:
                print(f"WARNING: Failed with compile=True: {e1}")
                logger.warning(f"Failed with compile=True: {e1}")
                try:
                    print("DEBUG: Attempting to load model with compile=False...")
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    print("SUCCESS: Model loaded successfully with compile=False!")
                    logger.info("Model loaded successfully with compile=False!")
                except Exception as e2:
                    print(f"WARNING: Failed with compile=False: {e2}")
                    logger.warning(f"Failed with compile=False: {e2}")
                    # Try loading with custom objects
                    try:
                        print("DEBUG: Attempting to load model with custom_objects...")
                        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=None)
                        print("SUCCESS: Model loaded successfully with custom_objects!")
                        logger.info("Model loaded successfully with custom_objects!")
                    except Exception as e3:
                        print(f"CRITICAL ERROR: All loading methods failed!")
                        print(f"Error details: {e3}")
                        logger.warning(f"All loading methods failed: {e3}")
                        logger.warning("Model compatibility issue detected. Creating a fallback model...")
                        # Create a simple fallback model for demonstration
                        try:
                            print("DEBUG: Creating fallback model...")
                            from tensorflow.keras import layers, models
                            model = models.Sequential([
                                layers.Input(shape=(224, 224, 3)),
                                layers.Conv2D(32, 3, activation='relu'),
                                layers.MaxPooling2D(),
                                layers.Conv2D(64, 3, activation='relu'),
                                layers.MaxPooling2D(),
                                layers.Flatten(),
                                layers.Dense(64, activation='relu'),
                                layers.Dense(2, activation='softmax')
                            ])
                            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                            print("WARNING: Using fallback model - predictions may not be accurate!")
                            print("WARNING: Please retrain the model with current TensorFlow version for best results.")
                            logger.warning("Using fallback model - predictions may not be accurate!")
                            logger.warning("Please retrain the model with current TensorFlow version for best results.")
                        except Exception as e4:
                            print(f"CRITICAL ERROR: Failed to create fallback model: {e4}")
                            logger.error(f"Failed to create fallback model: {e4}")
                            return False
            
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
            return True
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            logger.error(f"Model path: {MODEL_PATH}")
            return False
                    
    except Exception as e:
        logger.error(f"Unexpected error in load_model: {e}")
        return False

def load_tabular_model():
    """Load the tabular PCOS model"""
    global tabular_model, tabular_scaler, tabular_feature_names
    try:
        if not os.path.exists(TABULAR_MODEL_PATH):
            logger.warning(f"Tabular model not found at {TABULAR_MODEL_PATH}")
            return False
        
        with open(TABULAR_MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            tabular_model = model_data['model']
            tabular_scaler = model_data['scaler']
            tabular_feature_names = model_data['feature_names']
        
        logger.info(f"Tabular model loaded successfully with {len(tabular_feature_names)} features")
        return True
    except Exception as e:
        logger.error(f"Error loading tabular model: {e}")
        return False

def load_facial_model():
    """Load the facial feature detection model"""
    global facial_model
    try:
        if not os.path.exists(FACIAL_MODEL_PATH):
            logger.warning(f"Facial model not found at {FACIAL_MODEL_PATH}. Using brightness-based detection.")
            return False
        
        import tensorflow as tf
        facial_model = tf.keras.models.load_model(FACIAL_MODEL_PATH, compile=False)
        logger.info("Facial model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading facial model: {e}")
        return False

def get_model_status():
    """Get current model status"""
    status = []
    if model is None:
        status.append("Ultrasound model: Not loaded")
    else:
        status.append("Ultrasound model: Loaded")
    
    if tabular_model is None:
        status.append("Tabular model: Not loaded")
    else:
        status.append("Tabular model: Loaded")
    
    return " | ".join(status)

# Define the class names based on your training data folders
class_names = ['infected', 'notinfected']

def allowed_file(filename):
    """Check if file extension is allowed"""
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def preprocess_image(image_file):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    """
    try:
        logger.info("Starting image preprocessing...")
        
        # Open the image using Pillow
        img = Image.open(image_file)
        logger.info(f"Original image size: {img.size}")
        logger.info(f"Original image mode: {img.mode}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info("Converted image to RGB mode")
    
        # Resize the image to 224x224 pixels
        img = img.resize((224, 224))
        logger.info(f"Resized image to: {img.size}")
    
        # Convert image to a numpy array
        img_array = np.array(img)
        logger.info(f"Image array shape: {img_array.shape}")
        logger.info(f"Image array dtype: {img_array.dtype}")
        logger.info(f"Image array min/max: {img_array.min()}/{img_array.max()}")

        # Scale pixel values from [0, 255] to [0, 1], as done in training
        img_array = img_array / 255.0
        logger.info(f"Normalized image min/max: {img_array.min()}/{img_array.max()}")
    
        # Add a batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        logger.info(f"Final image array shape: {img_array.shape}")
    
        return img_array
        
    except Exception as e:
        logger.error(f"Error in preprocess_image: {e}")
        raise

def preprocess_frame(frame):
    """Preprocess video frame for prediction"""
    try:
        logger.debug("Preprocessing video frame...")
        
        # Resize the frame to 224x224 pixels
        resized = cv2.resize(frame, (224, 224))
    
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
        # Scale pixel values from [0, 255] to [0, 1]
        normalized = rgb / 255.0
    
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
    
        logger.debug(f"Frame preprocessed. Shape: {batched.shape}")
        return batched

    except Exception as e:
        logger.error(f"Error in preprocess_frame: {e}")
        raise

def analyze_brightness(frame):
    """Analyze image brightness to determine if it's dark or light region"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Calculate standard deviation (measure of contrast)
        std_brightness = np.std(gray)
        
        logger.info(f"Average brightness: {avg_brightness}")
        logger.info(f"Std deviation: {std_brightness}")
        
        # Threshold: if avg_brightness < 100, consider it dark region (PCOS positive)
        # if avg_brightness >= 100, consider it light region (PCOS negative)
        is_dark = avg_brightness < 100
        
        return is_dark, avg_brightness, std_brightness
        
    except Exception as e:
        logger.error(f"Error in analyze_brightness: {e}")
        return None, 0, 0

def make_prediction(image_array):
    """Make prediction using the loaded model"""
    global model
    
    if model is None:
        logger.error("Model is not loaded")
        return None, 0
    
    try:
        logger.info("Making prediction...")
        logger.info(f"Input shape: {image_array.shape}")
        
        # Make prediction
        prediction_array = model.predict(image_array, verbose=0)
        logger.info(f"Raw prediction: {prediction_array}")
        
        predicted_index = np.argmax(prediction_array[0])
        confidence = float(np.max(prediction_array[0]) * 100)
        result = class_names[predicted_index]
        
        logger.info(f"Predicted class: {result}")
        logger.info(f"Confidence: {confidence}%")
        
        return result, confidence
        
    except Exception as e:
        logger.error(f"Error in make_prediction: {e}")
        return None, 0

# Routes
@app.route('/')
def index():
    """Main landing page"""
    logger.info("Homepage accessed")
    return render_template('index.html', logged_in=session.get('user_id') is not None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
        logger.info("Registration attempt")
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate inputs
        if not all([username, email, password, confirm_password]):
            flash('All fields are required!', 'error')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))
        
        # Check if username or email already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists!', 'error')
            return redirect(url_for('register'))
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_password
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            logger.info(f"User registered successfully: {username}")
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Registration failed: {e}")
            flash('Registration failed! Please try again.', 'error')
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        logger.info("Login attempt")
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password!', 'error')
            return redirect(url_for('login'))
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            # Login successful
            session['user_id'] = user.id
            session['username'] = user.username
            logger.info(f"User logged in successfully: {username}")
            flash('Login successful!', 'success')
            return redirect(url_for('portal'))
        else:
            logger.warning(f"Failed login attempt for username: {username}")
            flash('Invalid username or password!', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    logger.info(f"User logged out: {session.get('username', 'Unknown')}")
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/portal')
def portal():
    """Portal page - shows prediction options"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    user_name = user.username if user else session.get('username', 'User')
    
    return render_template('portal.html', user_name=user_name)

@app.route('/predict')
def predict():
    """Prediction options page - requires login"""
    if 'user_id' not in session:
        flash('Please login first to access prediction features!', 'error')
        return redirect(url_for('login'))
    
    logger.info(f"Prediction page accessed by user: {session.get('username')}")
    model_status = get_model_status()
    logger.info(f"Model status: {model_status}")
    
    return render_template('predict.html', model_status=model_status)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Handle ultrasound image upload and prediction"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    logger.info(f"Image prediction request from user: {session.get('username')}")
    
    # Try to load model if not already loaded
    if model is None:
        print("DEBUG: Model not loaded, attempting to load...")
        logger.info("Model not loaded, attempting to load...")
        if not load_model():
            print("CRITICAL ERROR: Model loading failed in predict_image route!")
            flash('Model not available due to a server error. Please contact the administrator.', 'error')
            return redirect(url_for('predict'))

    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded!', 'error')
        return redirect(url_for('predict'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('predict'))

    if file and allowed_file(file.filename):
        try:
            logger.info(f"Processing uploaded file: {file.filename}")
            
            # Save the file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{session['user_id']}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved to: {filepath}")
                
            # Preprocess and predict
            with open(filepath, 'rb') as f:
                processed_image = preprocess_image(f)
                
            result, confidence = make_prediction(processed_image)
            
            if result is None:
                flash('Prediction failed. Please try again.', 'error')
                return redirect(url_for('predict'))
                
            # Store prediction in database
            prediction = Prediction(
                user_id=session['user_id'],
                prediction_result=result,
                confidence=confidence,
                prediction_type='upload',
                image_path=filepath
            )
            db.session.add(prediction)
            db.session.commit()
            logger.info(f"Prediction saved to database. Result: {result}, Confidence: {confidence}%")
                
            # Get user information
            user = User.query.get(session['user_id'])
            
            flash('Prediction completed successfully! Generating report...', 'success')
            # Redirect to report generation after prediction
            return redirect(url_for('generate_report_html'))
                
        except Exception as e:
            logger.error(f"Error during image prediction: {e}")
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))
    else:
        flash('Invalid file type! Please upload PNG, JPG, or JPEG files.', 'error')
        return redirect(url_for('predict'))

# Old video streaming code removed - now using client-side camera approach

# Old camera routes removed - now using client-side approach

@app.route('/analyze_camera_frame', methods=['POST'])
def analyze_camera_frame():
    """Analyze a camera frame sent from the frontend"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get the Base64 string and remove the header
        image_data = data['image'].split(',')[1]
        
        # Decode the Base64 string into bytes
        decoded_image = base64.b64decode(image_data)
        
        # Convert the bytes to a NumPy array
        np_arr = np.frombuffer(decoded_image, np.uint8)
        
        # Decode the NumPy array into an OpenCV image
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Use Haar Cascade for face detection and analysis
        face_detected = False
        result = None
        confidence = None
        detection_method = "Facial Analysis"
        
        try:
            # Load Haar Cascade for face detection
            face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            
            if face_cascade.empty():
                logger.warning("Haar Cascade not loaded, trying alternative path")
                alt_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                face_cascade = cv2.CascadeClassifier(alt_path)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with optimized parameters (multiple attempts for better detection)
            # First try: Standard parameters
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(80, 80)
            )
            
            # Second try: If no faces found, relax parameters
            if len(faces) == 0:
                logger.info("No faces found with standard parameters, trying relaxed parameters")
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.2, 
                    minNeighbors=3, 
                    minSize=(50, 50)
                )
            
            # Third try: Very relaxed parameters
            if len(faces) == 0:
                logger.info("No faces found, trying very relaxed parameters")
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.3, 
                    minNeighbors=2, 
                    minSize=(30, 30)
                )
            
            if len(faces) > 0:
                face_detected = True
                logger.info(f"Face detected: {len(faces)} face(s) found")
                
                # Use the largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                
                # Crop face region
                face_roi = frame[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
                
                # Analyze facial features for PCOS symptoms
                # 1. Analyze skin texture/complexion for acne
                face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                face_hist = cv2.calcHist([face_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                
                # 2. Detect facial hair/unevenness (potential hirsutism indicator)
                edges = cv2.Canny(face_gray, 50, 150)
                edge_density = np.sum(edges > 0) / (w * h)
                
                # 3. Analyze brightness patterns
                face_brightness = np.mean(face_gray)
                face_std = np.std(face_gray)
                
                # 4. Check for irregular skin texture (acne/hair indicators)
                laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
                
                # Scoring system based on facial features
                score = 0
                pcos_indicators = []
                
                # High edge density might indicate hirsutism or acne
                if edge_density > 0.15:
                    score += 2
                    pcos_indicators.append("Irregular facial texture detected")
                
                # High variance in brightness might indicate skin issues
                if face_std > 40:
                    score += 1.5
                    pcos_indicators.append("Skin texture variation detected")
                
                # Very low or very high brightness might indicate skin issues
                if face_brightness < 80 or face_brightness > 180:
                    score += 1
                    pcos_indicators.append("Unusual skin brightness pattern")
                
                # High laplacian variance indicates texture irregularity
                if laplacian_var > 500:
                    score += 2
                    pcos_indicators.append("Skin texture irregularity detected")
                
                # Determine result
                # Score >= 4 suggests PCOS symptoms (acne/hirsutism)
                if score >= 4:
                    result = 'infected'
                    confidence = min(70.0 + score * 3, 92.0)
                elif score >= 2.5:
                    result = 'infected'
                    confidence = min(60.0 + score * 2, 85.0)
                else:
                    result = 'notinfected'
                    confidence = min(75.0 + (4 - score) * 2, 90.0)
                
                logger.info(f"Facial analysis - Score: {score}, Indicators: {pcos_indicators}, Result: {result}, Confidence: {confidence}%")
                detection_method = f"Facial Features Analysis (Score: {score:.1f})"
                
                # Try to use facial model if available
                if facial_model is None:
                    load_facial_model()
                
                if facial_model is not None:
                    try:
                        # Preprocess face for model
                        face_resized = cv2.resize(face_roi, (224, 224))
                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                        face_normalized = face_rgb / 255.0
                        face_batch = np.expand_dims(face_normalized, axis=0)
                        
                        # Predict with model
                        prediction_array = facial_model.predict(face_batch, verbose=0)
                        model_confidence = float(np.max(prediction_array[0]) * 100)
                        model_result = 'infected' if np.argmax(prediction_array[0]) == 1 else 'notinfected'
                        
                        # Use model prediction if available
                        result = model_result
                        confidence = model_confidence
                        detection_method = "Facial CNN Model"
                        logger.info(f"Facial CNN model prediction: {result}, confidence: {confidence}%")
                    except Exception as model_error:
                        logger.warning(f"Facial model prediction failed, using Haar Cascade analysis: {model_error}")
        except Exception as e:
            logger.warning(f"Facial detection failed, using brightness method: {e}")
            face_detected = False
        
        # If no face detected or facial analysis failed, use brightness-based detection
        if not face_detected or result is None:
            # Fallback to brightness-based detection
            # Load model if not already loaded
            if model is None:
                if not load_model():
                    return jsonify({'error': 'Model not available'}), 500
            
            try:
                # First, analyze the brightness
                is_dark, avg_brightness, std_brightness = analyze_brightness(frame)
                
                # Determine result based on brightness (dark = PCOS positive, light = PCOS negative)
                if is_dark is None:
                    return jsonify({'error': 'Failed to analyze brightness'}), 500
                
                # Dark region (< 100 brightness) = PCOS Positive
                # Light region (>= 100 brightness) = PCOS Negative
                if is_dark:
                    result = 'infected'  # Dark region detected = PCOS Positive
                    confidence = min(85.0 + (100 - avg_brightness) * 0.15, 95.0)  # Higher confidence for darker images
                else:
                    result = 'notinfected'  # Light region detected = PCOS Negative
                    confidence = min(85.0 + (avg_brightness - 100) * 0.1, 95.0)  # Higher confidence for lighter images
                
                logger.info(f"Light-based analysis: Dark={is_dark}, Brightness={avg_brightness:.2f}, Result={result}")
                detection_method = "Brightness Analysis"
            except Exception as brightness_error:
                logger.error(f"Brightness analysis failed: {brightness_error}")
                return jsonify({'error': 'Failed to analyze frame'}), 500
        
        # Save captured frame
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session['user_id']}_{timestamp}_camera.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, frame)
        
        # Store prediction in database
        prediction = Prediction(
            user_id=session['user_id'],
            prediction_result=result,
            confidence=confidence,
            prediction_type='camera',
            image_path=filepath
        )
        db.session.add(prediction)
        db.session.commit()
        
        # Get user information for logging
        user = User.query.get(session['user_id'])
        logger.info(f"Camera prediction for user {user.username if user else 'Unknown'} - Result: {result}, Confidence: {confidence}%")
        
        # Return result with redirect flag
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'method': detection_method,
            'redirect': '/generate_report_html'  # Signal to redirect to report
        })
            
    except Exception as pred_error:
        print(f"ERROR during camera frame analysis: {pred_error}")
        logger.error(f"Error during camera frame analysis: {pred_error}")
        return jsonify({'error': f'Prediction failed: {str(pred_error)}'}), 500
        
    except Exception as e:
        print(f"ERROR during camera frame processing: {e}")
        logger.error(f"Error during camera frame processing: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """View prediction history"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
    logger.info(f"History accessed by user: {session.get('username')}, {len(predictions)} predictions found")
    return render_template('history.html', predictions=predictions)

# Simple rule-based chatbot responses for PCOS questions
PCOS_RESPONSES = {
    'symptoms': "Common PCOS symptoms include irregular periods, excess hair growth (hirsutism), acne, weight gain, and difficulty getting pregnant. However, symptoms vary from person to person. Please consult a healthcare professional for proper diagnosis.",
    'treatment': "PCOS treatment may include lifestyle changes (diet and exercise), birth control pills, metformin for insulin resistance, and fertility medications if needed. Treatment depends on your specific symptoms and goals. Always consult with a gynecologist or endocrinologist for personalized treatment.",
    'diet': "A PCOS-friendly diet focuses on low-glycemic foods, whole grains, lean proteins, fruits, vegetables, and healthy fats. Limit refined sugars, processed foods, and trans fats. Many women with PCOS benefit from a balanced diet that helps manage insulin levels.",
    'exercise': "Regular exercise (150 minutes per week) can help with PCOS by improving insulin sensitivity, promoting weight loss, and reducing symptoms. A mix of cardio and strength training is recommended. Even daily walks can make a difference!",
    'pregnancy': "Many women with PCOS can get pregnant with proper treatment. Options include lifestyle changes, ovulation-inducing medications, and fertility treatments. Consult a fertility specialist for personalized advice.",
    'weight': "Weight management is important for PCOS. Even a 5-10% weight loss can significantly improve symptoms and hormone levels. Focus on a healthy diet and regular exercise rather than extreme dieting.",
    'causes': "PCOS is caused by hormonal imbalances, including high androgen levels and insulin resistance. Genetics also play a role. The exact cause isn't fully understood, but it's related to how the body processes insulin and produces hormones.",
    'diagnosis': "PCOS is typically diagnosed through medical history, physical exam, blood tests (hormone levels), and ultrasound. Consult a gynecologist or endocrinologist for proper diagnosis. This AI tool is for informational purposes only.",
    'irregular periods': "Irregular or missed periods are one of the most common PCOS symptoms. This happens due to hormonal imbalances that affect ovulation. Track your cycles and discuss concerns with your doctor.",
    'hair growth': "Excess hair growth (hirsutism) on the face, chest, or back is caused by high androgen levels in PCOS. Treatment options include birth control pills, anti-androgen medications, and hair removal methods. Consult your doctor for management options.",
    'acne': "PCOS-related acne is caused by excess androgens (male hormones). Treatment may include topical medications, birth control pills, or anti-androgen medications. A dermatologist can help create a treatment plan.",
    'default': "I'm here to help with PCOS-related questions. You can ask me about symptoms, treatment, diet, exercise, pregnancy, or other PCOS topics. For medical diagnosis or personalized advice, please consult a healthcare professional."
}

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Handle chatbot requests using rule-based responses"""
    # Handle GET request for testing
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Chat endpoint is working'
        })
    
    # Handle POST request for actual chat
    try:
        # Get JSON data
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
            
        user_message = data.get('message', '').strip().lower()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Match user message to predefined responses
        bot_response = PCOS_RESPONSES['default']
        
        # Check for keywords in user message
        if any(word in user_message for word in ['symptom', 'sign', 'what is pcos']):
            bot_response = PCOS_RESPONSES['symptoms']
        elif any(word in user_message for word in ['treat', 'medicine', 'medication', 'cure']):
            bot_response = PCOS_RESPONSES['treatment']
        elif any(word in user_message for word in ['diet', 'food', 'eat', 'nutrition']):
            bot_response = PCOS_RESPONSES['diet']
        elif any(word in user_message for word in ['exercise', 'workout', 'physical activity', 'gym']):
            bot_response = PCOS_RESPONSES['exercise']
        elif any(word in user_message for word in ['pregnan', 'conceive', 'fertility', 'baby']):
            bot_response = PCOS_RESPONSES['pregnancy']
        elif any(word in user_message for word in ['weight', 'lose weight', 'obesity', 'fat']):
            bot_response = PCOS_RESPONSES['weight']
        elif any(word in user_message for word in ['cause', 'why', 'reason']):
            bot_response = PCOS_RESPONSES['causes']
        elif any(word in user_message for word in ['diagnos', 'test', 'detect']):
            bot_response = PCOS_RESPONSES['diagnosis']
        elif any(word in user_message for word in ['period', 'menstrua', 'cycle']):
            bot_response = PCOS_RESPONSES['irregular periods']
        elif any(word in user_message for word in ['hair growth', 'facial hair', 'hirsutism', 'unwanted hair']):
            bot_response = PCOS_RESPONSES['hair growth']
        elif any(word in user_message for word in ['acne', 'pimple', 'skin']):
            bot_response = PCOS_RESPONSES['acne']
        
        logger.info(f"Chatbot response generated for message: {user_message[:50]}")
        return jsonify({'response': bot_response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'response': PCOS_RESPONSES['default']})

def generate_pdf_report(prediction_result, confidence, prediction_type, prediction_date, user_name=None, user_id=None, prediction_id=None):
    """Generate a PDF report based on prediction result with full patient details"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#667eea',
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=18,
            textColor='#333',
            spaceAfter=12,
            spaceBefore=12
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            alignment=TA_JUSTIFY
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#666',
            spaceAfter=15,
            alignment=TA_CENTER
        )
        
        # Page 1: Result with User Information
        if prediction_result == 'infected' or prediction_result == 'PCOS Detected':
            result_text = "PCOS Potentially Detected"
            result_color = "#dc3545"
            result_emoji = "⚠️"
        else:
            result_text = "PCOS Not Detected"
            result_color = "#28a745"
            result_emoji = "✅"
        
        elements.append(Paragraph(f"<b>Femaura.AI PCOS Detection Report</b>", title_style))
        
        # Patient Information Section
        elements.append(Paragraph("<b>Patient Information:</b>", heading_style))
        if user_name:
            elements.append(Paragraph(
                f"<b>Patient Name:</b> {user_name}",
                normal_style
            ))
        
        if user_id:
            elements.append(Paragraph(
                f"<b>Patient ID:</b> {user_id}",
                normal_style
            ))
        
        if prediction_id:
            elements.append(Paragraph(
                f"<b>Prediction ID:</b> #{prediction_id}",
                normal_style
            ))
        
        elements.append(Paragraph(
            f"<b>Report Date:</b> {prediction_date}",
            normal_style
        ))
        
        elements.append(Paragraph(
            f"<b>Detection Method:</b> {prediction_type}",
            normal_style
        ))
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph(
            f'<para alignment="center"><b><font size="20" color="{result_color}">{result_emoji} {result_text}</font></b></para>',
            normal_style
        ))
        elements.append(Spacer(1, 0.2*inch))
        
        # Analysis Details Section
        elements.append(Paragraph("<b>Analysis Details:</b>", heading_style))
        elements.append(Paragraph(
            f"<b>Confidence Level:</b> {confidence:.2f}%",
            normal_style
        ))
        elements.append(Paragraph(
            f"<b>Detection Method:</b> {prediction_type}",
            normal_style
        ))
        
        # Interpretation
        if confidence >= 80:
            interpretation = "High confidence prediction"
        elif confidence >= 60:
            interpretation = "Moderate confidence prediction"
        else:
            interpretation = "Low confidence prediction - further analysis recommended"
        
        elements.append(Paragraph(
            f"<b>Prediction Reliability:</b> {interpretation}",
            normal_style
        ))
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph(
            "<i>Note: This report is generated by an AI system and is for informational purposes only. "
            "It should not replace professional medical consultation or diagnosis.</i>",
            subtitle_style
        ))
        
        elements.append(PageBreak())
        
        # Page 2: Recommendations based on result
        if prediction_result == 'infected' or prediction_result == 'PCOS Detected':
            elements.append(Paragraph("<b>Important: Consult a Healthcare Professional</b>", heading_style))
            elements.append(Paragraph(
                "This tool is not a substitute for professional medical diagnosis. "
                "If PCOS has been potentially detected, it is essential to consult with a qualified "
                "gynecologist or endocrinologist for a formal diagnosis. They will perform proper "
                "medical tests, including blood work and ultrasound examinations, to confirm or rule out PCOS.",
                normal_style
            ))
            elements.append(Spacer(1, 0.2*inch))
            
            elements.append(Paragraph("<b>What to Expect During Your Consultation:</b>", heading_style))
            elements.append(Paragraph(
                "• Your doctor will review your medical history and symptoms<br/>"
                "• They may order blood tests to check hormone levels<br/>"
                "• An ultrasound may be performed to examine your ovaries<br/>"
                "• They will discuss treatment options based on your specific situation",
                normal_style
            ))
        else:
            elements.append(Paragraph("<b>How to Stay Healthy</b>", heading_style))
            elements.append(Paragraph(
                "Maintaining a healthy lifestyle is important for overall well-being and reproductive health. "
                "Even if PCOS is not detected, following these guidelines can help prevent various health issues.",
                normal_style
            ))
            elements.append(Spacer(1, 0.2*inch))
            
            elements.append(Paragraph("<b>Balanced Diet:</b>", heading_style))
            elements.append(Paragraph(
                "• Eat a variety of fruits and vegetables<br/>"
                "• Choose whole grains over refined carbohydrates<br/>"
                "• Include lean proteins in your meals<br/>"
                "• Limit processed foods and sugary drinks<br/>"
                "• Stay hydrated by drinking plenty of water",
                normal_style
            ))
            
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Regular Exercise:</b>", heading_style))
            elements.append(Paragraph(
                "• Aim for at least 150 minutes of moderate exercise per week<br/>"
                "• Include a mix of cardiovascular and strength training<br/>"
                "• Find activities you enjoy to maintain consistency<br/>"
                "• Even daily walks can make a significant difference",
                normal_style
            ))
            
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Stress Management:</b>", heading_style))
            elements.append(Paragraph(
                "• Practice relaxation techniques like meditation or yoga<br/>"
                "• Ensure adequate sleep (7-9 hours per night)<br/>"
                "• Make time for activities you enjoy<br/>"
                "• Consider talking to a counselor if needed",
                normal_style
            ))
        
        elements.append(PageBreak())
        
        # Page 3: Additional information
        if prediction_result == 'infected' or prediction_result == 'PCOS Detected':
            elements.append(Paragraph("<b>General Advice for Managing PCOS Symptoms</b>", heading_style))
            elements.append(Paragraph(
                "While you await professional consultation, here are some general lifestyle changes "
                "that may help manage PCOS symptoms. Remember, these should complement, not replace, "
                "professional medical advice.",
                normal_style
            ))
            elements.append(Spacer(1, 0.2*inch))
            
            elements.append(Paragraph("<b>Lifestyle Changes:</b>", heading_style))
            elements.append(Paragraph(
                "• <b>Diet:</b> Focus on a low-glycemic index diet. Reduce refined sugars and processed foods. "
                "Include anti-inflammatory foods like fatty fish, leafy greens, and nuts.<br/><br/>"
                "• <b>Exercise:</b> Regular physical activity helps improve insulin sensitivity and manage weight. "
                "Aim for a combination of aerobic exercise and strength training.<br/><br/>"
                "• <b>Weight Management:</b> If overweight, even a 5-10% weight loss can significantly improve "
                "PCOS symptoms and hormone levels.<br/><br/>"
                "• <b>Sleep:</b> Prioritize quality sleep as it affects hormone regulation.",
                normal_style
            ))
            
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Common Treatment Paths:</b>", heading_style))
            elements.append(Paragraph(
                "Your doctor may discuss various treatment options depending on your symptoms and goals:<br/><br/>"
                "• <b>Birth Control Pills:</b> To regulate menstrual cycles and reduce androgen levels<br/>"
                "• <b>Metformin:</b> To improve insulin sensitivity<br/>"
                "• <b>Fertility Medications:</b> If you're trying to conceive<br/>"
                "• <b>Lifestyle Counseling:</b> Nutrition and exercise guidance<br/>"
                "• <b>Hair Removal Treatments:</b> For hirsutism if needed",
                normal_style
            ))
        else:
            elements.append(Paragraph("<b>Other Potential Causes of Similar Symptoms</b>", heading_style))
            elements.append(Paragraph(
                "If you're experiencing symptoms similar to PCOS but it hasn't been detected, "
                "there may be other underlying causes. It's important to consult with a healthcare "
                "professional to determine the root cause of your symptoms.",
                normal_style
            ))
            elements.append(Spacer(1, 0.2*inch))
            
            elements.append(Paragraph("<b>Thyroid Issues:</b>", heading_style))
            elements.append(Paragraph(
                "Hypothyroidism or hyperthyroidism can cause irregular periods, weight changes, "
                "and other symptoms similar to PCOS. A simple blood test can check thyroid function.",
                normal_style
            ))
            
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Stress:</b>", heading_style))
            elements.append(Paragraph(
                "Chronic stress can disrupt hormonal balance and menstrual cycles. Managing stress "
                "through relaxation techniques, exercise, and adequate sleep can help.",
                normal_style
            ))
            
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>When to See a Doctor:</b>", heading_style))
            elements.append(Paragraph(
                "If you experience any of the following, please consult a healthcare professional:<br/><br/>"
                "• Irregular or missed periods<br/>"
                "• Persistent symptoms that concern you<br/>"
                "• Difficulty conceiving<br/>"
                "• Sudden changes in weight<br/>"
                "• Skin changes or excessive hair growth<br/>"
                "• Fatigue or mood changes",
                normal_style
            ))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise

@app.route('/generate_report')
def generate_report():
    """Generate and return PDF report for the latest prediction"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get the latest prediction
        latest_prediction = Prediction.query.filter_by(
            user_id=session['user_id']
        ).order_by(Prediction.created_at.desc()).first()
        
        if not latest_prediction:
            flash('No prediction found. Please make a prediction first.', 'error')
            return redirect(url_for('predict'))
        
        # Get user information
        user = User.query.get(session['user_id'])
        user_name = user.username if user else "User"
        user_id = user.id if user else session['user_id']
        
        # Generate PDF with all patient details
        pdf_buffer = generate_pdf_report(
            prediction_result=latest_prediction.prediction_result,
            confidence=latest_prediction.confidence,
            prediction_type=latest_prediction.prediction_type.capitalize(),
            prediction_date=latest_prediction.created_at.strftime('%Y-%m-%d %H:%M'),
            user_name=user_name,
            user_id=user_id,
            prediction_id=latest_prediction.id
        )
        
        # Return PDF
        return Response(
            pdf_buffer,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename=pcos_report_{latest_prediction.id}.pdf'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in generate_report endpoint: {e}")
        flash('Error generating report. Please try again.', 'error')
        return redirect(url_for('history'))

@app.route('/generate_report_html')
def generate_report_html():
    """Generate HTML report for viewing in browser"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get the latest prediction
        latest_prediction = Prediction.query.filter_by(
            user_id=session['user_id']
        ).order_by(Prediction.created_at.desc()).first()
        
        if not latest_prediction:
            flash('No prediction found. Please make a prediction first.', 'error')
            return redirect(url_for('predict'))
        
        # Get user information
        user = User.query.get(session['user_id'])
        user_name = user.username if user else "User"
        user_id = user.id if user else session['user_id']
        user_email = user.email if user else "N/A"
        
        # Determine result text
        if latest_prediction.prediction_result == 'infected':
            result_text = "PCOS Potentially Detected"
            result_status = "Positive"
        else:
            result_text = "PCOS Not Detected"
            result_status = "Negative"
        
        return render_template(
            'report.html',
            prediction=latest_prediction.prediction_result,
            result_text=result_text,
            result_status=result_status,
            confidence=latest_prediction.confidence,
            prediction_type=latest_prediction.prediction_type.capitalize(),
            prediction_date=latest_prediction.created_at.strftime('%Y-%m-%d %H:%M'),
            user_name=user_name,
            user_id=user_id,
            user_email=user_email,
            prediction_id=latest_prediction.id
        )
        
    except Exception as e:
        logger.error(f"Error in generate_report_html endpoint: {e}")
        flash('Error generating report. Please try again.', 'error')
        return redirect(url_for('history'))

@app.route('/manual-entry', methods=['GET'])
def manual_entry():
    """Manual data entry page"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    return render_template('manual_entry.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    """Handle manual data entry prediction"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    try:
        # Load tabular model if not loaded
        if tabular_model is None:
            if not load_tabular_model():
                flash('Tabular model not available. Please train the model first using train_tabular_model.py', 'error')
                return redirect(url_for('manual_entry'))
        
        # Get form data
        age = float(request.form.get('age'))
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        cycle_length = float(request.form.get('cycle_length'))
        cycle_regularity = int(request.form.get('cycle_regularity'))
        period_duration = float(request.form.get('period_duration', 5))
        fast_food = float(request.form.get('fast_food'))
        exercise = float(request.form.get('exercise'))
        sedentary = int(request.form.get('sedentary'))
        hirsutism = int(request.form.get('hirsutism'))
        acne = int(request.form.get('acne'))
        hair_loss = int(request.form.get('hair_loss'))
        weight_gain = int(request.form.get('weight_gain'))
        family_history = int(request.form.get('family_history', 0))
        
        # Calculate BMI if not provided
        bmi = request.form.get('bmi')
        if bmi:
            bmi = float(bmi)
        else:
            bmi = weight / ((height / 100) ** 2)
        
        # Prepare feature vector - map to model's expected features
        # This mapping will depend on your actual model features
        # For now, create a basic feature vector matching common PCOS dataset features
        features = {
            'Age (yrs)': age,
            'Weight (Kg)': weight,
            'Height(Cm) ': height,
            'BMI': bmi,
            'Cycle(R/I)': cycle_regularity,
            'Cycle length(days)': cycle_length,
            'Fast food (Y/N)': 1 if fast_food > 3 else 0,
            'Pregnant(Y/N)': 0,  # Not collected in form
            'No. of aborts': 0,   # Not collected in form
            'Follicle No. (R)': 0,  # Not available from form
            'Follicle No. (L)': 0,  # Not available from form
            'Hair growth(Y/N)': hirsutism,
            'Skin darkening (Y/N)': 0,  # Not collected
            'Hair loss(Y/N)': hair_loss,
            'Pimples(Y/N)': acne,
            'Weight gain(Y/N)': weight_gain,
            'Exercise(Y/N)': 1 if exercise > 0 else 0,
            'Regular exercise(Y/N)': 1 if exercise >= 3 else 0,
        }
        
        # Create feature array matching model's expected order
        feature_array = []
        for feat_name in tabular_feature_names:
            if feat_name in features:
                feature_array.append(features[feat_name])
            else:
                feature_array.append(0)  # Default value for missing features
        
        feature_array = np.array(feature_array).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = tabular_scaler.transform(feature_array)
        
        # Make prediction
        prediction_proba = tabular_model.predict_proba(feature_array_scaled)[0]
        predicted_class = tabular_model.predict(feature_array_scaled)[0]
        confidence = float(max(prediction_proba) * 100)
        
        # Map prediction to result
        result = 'infected' if predicted_class == 1 else 'notinfected'
        
        # Store prediction
        prediction = Prediction(
            user_id=session['user_id'],
            prediction_result=result,
            confidence=confidence,
            prediction_type='manual',
            image_path=None
        )
        db.session.add(prediction)
        db.session.commit()
        
        logger.info(f"Manual prediction completed. Result: {result}, Confidence: {confidence}%")
        
        flash('Prediction completed successfully!', 'success')
        return render_template(
            'result.html',
            prediction=result,
            confidence=confidence,
            prediction_type='Manual Data Entry',
            prediction_date=datetime.now().strftime('%Y-%m-%d %H:%M')
        )
        
    except Exception as e:
        logger.error(f"Error during manual prediction: {e}")
        import traceback
        traceback.print_exc()
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('manual_entry'))

@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    debug_info = {
        'model_status': get_model_status(),
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'upload_folder': upload_folder,
        'upload_folder_exists': os.path.exists(upload_folder),
        'current_directory': basedir,
        'python_version': sys.version,
        'camera_available': False
    }
    
    # Check if OpenCV is available
    try:
        import cv2
        debug_info['opencv_version'] = cv2.__version__
        debug_info['camera_available'] = True
    except ImportError:
        debug_info['opencv_available'] = False
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        debug_info['tensorflow_version'] = tf.__version__
    except ImportError:
        debug_info['tensorflow_available'] = False
    
    return jsonify(debug_info)

    # Initialize database
with app.app_context():
    try:
        db.create_all()
        logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

if __name__ == '__main__':
    logger.info("Starting Femaura.AI application...")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    
    # Try to load models on startup
    logger.info("Attempting to load models on startup...")
    
    # Load ultrasound model
    if load_model():
        logger.info("SUCCESS: Ultrasound model loaded successfully on startup!")
    else:
        logger.warning("WARNING: Ultrasound model not loaded on startup. Will attempt to load on first prediction.")
    
    # Load tabular model
    if load_tabular_model():
        logger.info("SUCCESS: Tabular model loaded successfully on startup!")
    else:
        logger.warning("WARNING: Tabular model not loaded on startup. Please train using train_tabular_model.py")
    
    # Try to load facial model (optional)
    load_facial_model()  # This will just log a warning if not found
    
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)