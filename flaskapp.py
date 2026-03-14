from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify, Response
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow import keras
from keras.preprocessing.image import load_img
import numpy as np
import os
import cv2
from datetime import datetime
import base64

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# --- Database Configuration ---
# Connect to the 'femaura_db' database for Femaura.AI
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'femaura_db'

# Initialize MySQL
mysql = MySQL(app)

# --- Load the Machine Learning Model ---
# Load the pre-trained PCOS detection model
try:
    model = keras.models.load_model('model.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['PCOS Detected', 'PCOS Not Detected']

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Helper Functions ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_pcos(image_path):
    """
    Takes an image path, preprocesses the image, and returns the model's prediction.
    """
    if model is None:
        return "Model not loaded", 0
        
    try:
        # Load and preprocess the image for the model
        image = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)
        
        # Make a prediction
        prediction = model.predict(img_array)
        
        # Determine the class label from the prediction probabilities
        # Assuming binary classification with sigmoid output
        confidence = prediction[0][0]
        
        if confidence > 0.5:
            predicted_class = CLASS_NAMES[1]  # PCOS Not Detected
            confidence_percent = confidence * 100
        else:
            predicted_class = CLASS_NAMES[0]  # PCOS Detected
            confidence_percent = (1 - confidence) * 100
        
        return predicted_class, confidence_percent
    except Exception as e:
        return f"Prediction error: {e}", 0

def predict_pcos_frame(frame):
    """
    Takes a frame, preprocesses it, and returns the model's prediction.
    Used for real-time camera prediction.
    """
    if model is None:
        return "Model not loaded", 0
        
    try:
        # Preprocess the frame
        img_resized = cv2.resize(frame, IMAGE_SIZE)
        img_array = np.asarray(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Make a prediction
        prediction = model.predict(img_batch)[0][0]
        
        # Determine the class and confidence
        if prediction > 0.5:
            predicted_class = CLASS_NAMES[1]  # PCOS Not Detected
            confidence = prediction * 100
        else:
            predicted_class = CLASS_NAMES[0]  # PCOS Detected
            confidence = (1 - prediction) * 100
        
        return predicted_class, confidence
    except Exception as e:
        return f"Prediction error: {e}", 0

# --- Route Definitions ---
@app.route('/')
def index():
    """Renders the main landing page of the application."""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        phone = request.form.get('phone', '')
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        try:
            cursor = mysql.connection.cursor()
            # Check if email already exists
            cursor.execute("SELECT * FROM users WHERE email = %s", [email])
            existing_user = cursor.fetchone()
            
            if existing_user:
                flash("Email already registered. Please login.")
                cursor.close()
                return redirect(url_for('register'))
            
            # Insert new user
            sql_query = """
            INSERT INTO users (name, email, password, phone, created_at) 
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql_query, (name, email, hashed_password, phone, datetime.now()))
            mysql.connection.commit()
            cursor.close()
            
            flash("Registration successful! Please login.")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Registration error: {e}")
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/loginuser', methods=['POST'])
def loginuser():
    """Handle user login."""
    email = request.form['email']
    password = request.form['password']
    
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", [email])
        user = cursor.fetchone()
        cursor.close()
        
        if user and check_password_hash(user[3], password):  # user[3] is password column
            session['user_id'] = user[0]  # user[0] is id
            session['user_name'] = user[1]  # user[1] is name
            session['user_email'] = user[2]  # user[2] is email
            flash("Login successful!")
            return redirect(url_for('portal'))
        else:
            flash("Invalid email or password")
            return redirect(url_for('index'))
    except Exception as e:
        flash(f"Login error: {e}")
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Handle user logout."""
    session.clear()
    flash("Logged out successfully!")
    return redirect(url_for('index'))

@app.route('/forgetpass', methods=['GET', 'POST'])
def forgetpass():
    """Handle forgot password."""
    if request.method == 'POST':
        email = request.form['email']
        new_password = request.form['new_password']
        
        try:
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM users WHERE email = %s", [email])
            user = cursor.fetchone()
            
            if user:
                hashed_password = generate_password_hash(new_password)
                cursor.execute("UPDATE users SET password = %s WHERE email = %s", 
                             (hashed_password, email))
                mysql.connection.commit()
                cursor.close()
                flash("Password reset successful! Please login.")
                return redirect(url_for('index'))
            else:
                flash("Email not found")
                cursor.close()
                return redirect(url_for('forgetpass'))
        except Exception as e:
            flash(f"Error: {e}")
            return redirect(url_for('forgetpass'))
    
    return render_template('forgetpass.html')

@app.route('/portal')
def portal():
    """Renders the portal page where users can upload images or use camera."""
    if 'user_id' not in session:
        flash("Please login first")
        return redirect(url_for('index'))
    
    return render_template('portal.html', user_name=session.get('user_name'))

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission for PCOS prediction via image upload.
    It saves patient data and the prediction result to the database.
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    # Get patient data from the form
    patient_name = request.form.get('name', session.get('user_name'))
    age = request.form.get('age', '')
    phone_number = request.form.get('phone', '')
    symptoms = request.form.get('symptoms', '')
    
    # Check if file is present
    if 'file' not in request.files:
        flash("No file uploaded")
        return redirect(url_for('portal'))
    
    ultrasound_image = request.files['file']
    
    # Check if file is valid
    if ultrasound_image.filename == '':
        flash("No file selected")
        return redirect(url_for('portal'))
    
    if not allowed_file(ultrasound_image.filename):
        flash("Invalid file type. Please upload PNG, JPG, or JPEG images only.")
        return redirect(url_for('portal'))
    
    # --- Image Handling and Prediction ---
    try:
        # Save the uploaded image to the server
        filename = f"{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ultrasound_image.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        ultrasound_image.save(image_path)
        
        # Get the prediction result from the model
        prediction_result, confidence = predict_pcos(image_path)
        
        # --- Database Interaction ---
        cursor = mysql.connection.cursor()
        
        # Insert the prediction data
        sql_query = """
        INSERT INTO predictions (user_id, patient_name, age, phone_number, symptoms, 
                                prediction_result, confidence, image_path, prediction_type, created_at) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql_query, (session['user_id'], patient_name, age, phone_number, 
                                   symptoms, prediction_result, confidence, filename, 
                                   'upload', datetime.now()))
        mysql.connection.commit()
        cursor.close()
        
        # Flash a success message and show the result
        flash(f"Prediction Result: {prediction_result} (Confidence: {confidence:.2f}%)")
        return redirect(url_for('portal'))
    
    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for('portal'))

@app.route('/camera_predict', methods=['POST'])
def camera_predict():
    """
    Handles real-time camera prediction.
    Receives image data from the camera and returns prediction.
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        # Get the image data from request
        image_data = request.json.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Decode the base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get prediction
        prediction_result, confidence = predict_pcos_frame(frame)
        
        return jsonify({
            'prediction': prediction_result,
            'confidence': f"{confidence:.2f}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_camera_prediction', methods=['POST'])
def save_camera_prediction():
    """
    Saves the camera prediction to the database.
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    try:
        data = request.json
        prediction_result = data.get('prediction')
        confidence = data.get('confidence')
        symptoms = data.get('symptoms', '')
        
        cursor = mysql.connection.cursor()
        
        sql_query = """
        INSERT INTO predictions (user_id, patient_name, prediction_result, confidence, 
                                symptoms, prediction_type, created_at) 
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql_query, (session['user_id'], session['user_name'], 
                                   prediction_result, confidence, symptoms, 
                                   'camera', datetime.now()))
        mysql.connection.commit()
        cursor.close()
        
        return jsonify({'success': True, 'message': 'Prediction saved successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """View prediction history."""
    if 'user_id' not in session:
        flash("Please login first")
        return redirect(url_for('index'))
    
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("""
            SELECT prediction_result, confidence, prediction_type, symptoms, created_at 
            FROM predictions 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, [session['user_id']])
        predictions = cursor.fetchall()
        cursor.close()
        
        return render_template('history.html', predictions=predictions)
    except Exception as e:
        flash(f"Error: {e}")
        return redirect(url_for('portal'))

# --- Main Application Runner ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
