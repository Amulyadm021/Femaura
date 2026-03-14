# Femaura.AI - PCOS Detection Application

Femaura.AI is a comprehensive web application that uses artificial intelligence to detect Polycystic Ovary Syndrome (PCOS) through ultrasound image analysis. The application provides both image upload functionality and real-time camera detection capabilities.

## Features

### 🔐 User Authentication
- User registration and login system
- Secure password hashing
- Session management
- Protected prediction routes

### 🖼️ PCOS Detection Methods
1. **Image Upload**: Upload ultrasound images for analysis
2. **Real-time Camera**: Use live camera feed for instant detection
3. **Prediction Results**: Detailed analysis with confidence scores
4. **History Tracking**: View all previous predictions

### 🎯 Key Capabilities
- AI-powered PCOS detection using deep learning
- Support for multiple image formats (PNG, JPG, JPEG)
- Real-time video streaming with overlay predictions
- Comprehensive result reporting
- User-friendly interface with responsive design

## Technology Stack

### Backend
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **TensorFlow/Keras**: Machine learning model
- **OpenCV**: Computer vision and image processing
- **Pillow**: Image manipulation
- **SQLite**: Database

### Frontend
- **HTML5/CSS3**: Structure and styling
- **Bootstrap**: Responsive framework
- **JavaScript**: Interactive functionality
- **Jinja2**: Template engine

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd FightOS-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app_simple.py
   ```

4. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:5000`

## Usage

### Getting Started

1. **Register an Account**
   - Click "Register" on the homepage
   - Fill in username, email, and password
   - Confirm your password

2. **Login**
   - Use your credentials to log in
   - Access the prediction features

### Making Predictions

#### Method 1: Image Upload
1. Navigate to the "Predict" page
2. Click "Choose File" under "Upload Ultrasound Image"
3. Select your ultrasound image
4. Click "Analyze Image"
5. View the results

#### Method 2: Live Camera
1. On the "Predict" page, locate "Live Camera Detection"
2. The camera feed will start automatically
3. Position your ultrasound image in front of the camera
4. Click "Capture & Analyze"
5. View the real-time results

### Understanding Results

- **PCOS Detected**: Red indicator with confidence percentage
- **No PCOS Detected**: Green indicator with confidence percentage
- **Confidence Score**: Indicates the model's certainty (0-100%)

## File Structure

```
FightOS-main/
├── app.py                 # Main Flask application (with TensorFlow)
├── app_simple.py          # Simplified Flask application
├── model.h5              # Trained PCOS detection model
├── requirements.txt      # Python dependencies
├── users.db             # SQLite database (created on first run)
├── uploads/             # Directory for uploaded images
├── templates/           # HTML templates
│   ├── layout.html      # Base template
│   ├── index.html       # Homepage
│   ├── login.html       # Login page
│   ├── register.html    # Registration page
│   ├── predict.html     # Prediction interface
│   ├── result.html      # Results display
│   └── history.html     # Prediction history
└── static/              # Static assets
    ├── css/             # Stylesheets
    ├── js/              # JavaScript files
    └── img/             # Images and icons
```

## Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Encrypted password
- `created_at`: Registration timestamp

### Predictions Table
- `id`: Primary key
- `user_id`: Foreign key to users table
- `prediction_result`: PCOS detection result
- `confidence`: Confidence score
- `prediction_type`: Upload or camera method
- `image_path`: Path to uploaded image
- `created_at`: Prediction timestamp

## API Endpoints

### Authentication
- `GET/POST /register` - User registration
- `GET/POST /login` - User login
- `GET /logout` - User logout

### Predictions
- `GET /predict` - Prediction interface
- `POST /predict_image` - Image upload prediction
- `GET /video_feed` - Camera video stream
- `POST /capture_prediction` - Camera prediction
- `POST /stop_camera` - Stop camera stream

### Data
- `GET /history` - View prediction history

## Security Features

- Password hashing using Werkzeug
- Session-based authentication
- Protected routes requiring login
- File upload validation
- SQL injection prevention through SQLAlchemy ORM

## Model Information

The application uses a pre-trained deep learning model (`model.h5`) that:
- Was trained on ultrasound images
- Classifies images as "infected" (PCOS positive) or "notinfected" (PCOS negative)
- Expects 224x224 pixel RGB images
- Returns confidence scores for predictions

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure `model.h5` exists in the project directory
   - Check TensorFlow installation
   - Verify Python version compatibility

2. **Database Issues**
   - Delete `users.db` to reset the database
   - Ensure write permissions in the project directory

3. **Camera Not Working**
   - Check camera permissions in browser
   - Ensure camera is not used by other applications
   - Try refreshing the page

4. **Import Errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.10+ recommended)

## Disclaimer

**Important Medical Disclaimer**: This application is for educational and informational purposes only. The predictions generated by this AI model should not be considered as medical diagnosis or replace professional medical consultation. Always consult with qualified healthcare professionals for proper diagnosis and treatment of PCOS or any other medical conditions.

## Contributing

This is a demonstration project. For production use, consider:
- Implementing proper error handling
- Adding input validation
- Enhancing security measures
- Adding comprehensive testing
- Implementing proper logging

## License

This project is for educational purposes. Please ensure compliance with medical device regulations if used in clinical settings.

## Support

For technical support or questions, please contact the development team or refer to the documentation.

---

**Femaura.AI** - Empowering women's health through AI technology.


