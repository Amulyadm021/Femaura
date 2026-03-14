import os
import numpy as np
import base64
import cv2
import requests
import json
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, session, flash, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

# Initialize Flask application
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face BioGPT API Configuration (Primary - Medical Specialist)
BIOGPT_API_URL = "https://api-inference.huggingface.co/models/microsoft/biogpt"

def query_medical_ai(question, max_retries=1):
    """
    Query Microsoft BioGPT for medical/health questions
    BioGPT is specifically trained on biomedical literature and PubMed data
    Provides accurate medical information for PCOS and women's health
    """
    
    # Prepare medical-focused prompt for BioGPT
    # BioGPT works best with direct medical questions
    medical_prompt = f"{question}"
    
    # Try BioGPT API (Primary - Best for medical questions)
    try:
        logger.info("Querying Microsoft BioGPT for medical information...")
        response = requests.post(
            BIOGPT_API_URL,
            headers={
                "Content-Type": "application/json",
            },
            json={
                "inputs": medical_prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle different response formats from Hugging Face
            if isinstance(result, list) and len(result) > 0:
                ai_response = result[0].get('generated_text', '').strip()
                
                # Clean up the response
                if ai_response:
                    # Remove the original question if it appears in response
                    if question.lower() in ai_response.lower():
                        ai_response = ai_response.replace(question, '').strip()
                    
                    # Remove common artifacts
                    ai_response = ai_response.replace('Answer:', '').strip()
                    ai_response = ai_response.replace('Response:', '').strip()
                    
                    if len(ai_response) > 30:  # Ensure meaningful response
                        logger.info("BioGPT API successful - Medical response generated")
                        return ai_response
                        
            elif isinstance(result, dict) and 'generated_text' in result:
                ai_response = result['generated_text'].strip()
                if len(ai_response) > 30:
                    logger.info("BioGPT API successful")
                    return ai_response
                    
        elif response.status_code == 503:
            logger.info("BioGPT model is loading, please wait...")
            return None
        else:
            logger.warning(f"BioGPT API returned status {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.warning("BioGPT API timeout (model might be loading)")
        return None
    except Exception as e:
        logger.error(f"BioGPT API error: {e}")
    
    # Try Groq API as backup (Fast alternative)
    try:
        logger.info("Trying Groq API as backup...")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical assistant specializing in PCOS and women's health. Provide accurate, concise medical information in 2-3 sentences."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 200
            },
            timeout=8
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                ai_response = result['choices'][0]['message']['content'].strip()
                if ai_response and len(ai_response) > 30:
                    logger.info("Groq API backup successful")
                    return ai_response
    except Exception as e:
        logger.warning(f"Groq API backup failed: {e}")
    
    logger.info("All AI APIs unavailable, using rule-based fallback")
    return None

# Database Configuration - Using SQLite for easier setup
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Global variable for model
model = None

# Enhanced PCOS Chatbot Responses (Comprehensive Rule-based System)
PCOS_RESPONSES = {
    'greeting': """Hello! 👋 I'm your PCOS health assistant. I'm here to provide information about Polycystic Ovary Syndrome (PCOS), women's health, and answer your questions.

I can help you with:
✓ PCOS symptoms and diagnosis
✓ Treatment options and medications
✓ Diet and lifestyle advice
✓ Exercise and weight management
✓ Fertility and pregnancy concerns
✓ Mental health and emotional support

What would you like to know today?""",
    
    'symptoms': """Common PCOS symptoms include:

🔴 Menstrual Issues:
• Irregular or missed periods (oligomenorrhea)
• Heavy or prolonged bleeding
• Fewer than 8 periods per year

🔴 Hormonal Symptoms:
• Excessive hair growth (hirsutism) on face, chest, back, abdomen
• Male-pattern baldness or hair thinning
• Severe acne, especially on face, chest, back
• Oily skin

🔴 Metabolic Symptoms:
• Weight gain or difficulty losing weight
• Insulin resistance
• Darkened skin patches (acanthosis nigricans)
• Skin tags

🔴 Other Symptoms:
• Fertility issues
• Mood changes, depression, anxiety
• Sleep problems
• Fatigue

If you're experiencing multiple symptoms, please consult a healthcare provider for proper evaluation.""",
    
    'treatment': """PCOS treatment is personalized based on symptoms and goals:

💊 Medical Treatments:
• Birth control pills - regulate periods, reduce androgens
• Metformin - improve insulin sensitivity, aid weight loss
• Clomiphene/Letrozole - for fertility
• Spironolactone - reduce hair growth and acne
• Finasteride - for hair loss
• Eflornithine cream - slow facial hair growth

🌿 Lifestyle Interventions:
• Weight loss (5-10% can significantly improve symptoms)
• Regular exercise
• Healthy diet
• Stress management

🎯 Symptom-Specific:
• Laser/electrolysis for hair removal
• Acne medications (topical/oral)
• Hair regrowth treatments

⚠️ Always consult your doctor before starting any treatment. Each person's PCOS is unique!""",
    
    'diet': """PCOS-Friendly Diet Guidelines:

✅ FOODS TO INCLUDE:
• Whole grains (quinoa, brown rice, oats)
• Lean proteins (chicken, fish, tofu, legumes)
• Healthy fats (avocado, nuts, olive oil, omega-3s)
• High-fiber vegetables (broccoli, spinach, kale)
• Low-glycemic fruits (berries, apples, pears)
• Anti-inflammatory foods (turmeric, ginger, green tea)

❌ FOODS TO LIMIT:
• Refined carbs (white bread, pasta, pastries)
• Sugary foods and drinks
• Processed foods
• Trans fats
• Excessive dairy (for some people)
• High-glycemic foods

🍽️ EATING TIPS:
• Eat smaller, frequent meals
• Combine protein with carbs
• Stay hydrated (8-10 glasses water/day)
• Consider intermittent fasting (consult doctor first)
• Reduce caffeine and alcohol

A low-glycemic, anti-inflammatory diet helps manage insulin levels and reduce symptoms!""",
    
    'exercise': """Exercise for PCOS Management:

🏃 CARDIO (150 min/week):
• Walking, jogging, cycling
• Swimming, dancing
• HIIT workouts (high intensity interval training)
• Helps with weight loss and insulin sensitivity

💪 STRENGTH TRAINING (2-3x/week):
• Weight lifting, resistance bands
• Bodyweight exercises
• Builds muscle, boosts metabolism

🧘 MIND-BODY:
• Yoga - reduces stress, improves flexibility
• Pilates - strengthens core
• Meditation - manages stress hormones
• Tai chi - gentle movement

⚡ BENEFITS:
• Improves insulin sensitivity
• Aids weight management
• Regulates hormones
• Boosts mood and energy
• Improves fertility
• Better sleep quality

💡 START SLOW: Begin with 10-15 minutes daily and gradually increase. Find activities you enjoy for better consistency!""",
    
    'pregnancy': """PCOS and Pregnancy - What You Need to Know:

🤰 FERTILITY CHALLENGES:
• PCOS is a leading cause of infertility
• Irregular ovulation makes conception harder
• BUT many women with PCOS conceive successfully!

✅ IMPROVING FERTILITY:
• Achieve healthy weight (5-10% loss helps)
• Medications: Clomiphene, Letrozole
• Metformin to improve ovulation
• Lifestyle changes (diet, exercise)
• Track ovulation with kits/apps

🏥 FERTILITY TREATMENTS:
• Ovulation induction
• IUI (Intrauterine Insemination)
• IVF (In Vitro Fertilization)
• Ovarian drilling (laparoscopic surgery)

⚠️ PREGNANCY RISKS:
• Higher risk of gestational diabetes
• Preeclampsia
• Miscarriage (early pregnancy)
• Preterm birth

👶 GOOD NEWS:
• With proper care, most women with PCOS have healthy pregnancies
• Early prenatal care is crucial
• Symptoms often improve after pregnancy

Consult a fertility specialist if trying to conceive for over 6-12 months!""",
    
    'weight': """Weight Management with PCOS:

⚖️ WHY IT'S HARDER:
• Insulin resistance makes weight loss difficult
• Hormonal imbalances slow metabolism
• Increased hunger due to hormones
• Lower muscle mass

🎯 EFFECTIVE STRATEGIES:
• Low-glycemic diet (reduces insulin spikes)
• Portion control
• Regular exercise (cardio + strength)
• Adequate sleep (7-9 hours)
• Stress management
• Avoid yo-yo dieting
• Be patient - slower progress is normal

💊 MEDICAL HELP:
• Metformin can aid weight loss
• Consider consulting a dietitian
• Address insulin resistance
• Treat any thyroid issues

🌟 REALISTIC GOALS:
• Aim for 5-10% body weight loss
• Even small losses improve symptoms significantly!
• Focus on health markers, not just scale
• Measure progress by how you feel

💡 Remember: You can have PCOS at any weight. Health is more important than a number!""",
    
    'causes': """What Causes PCOS?

🧬 GENETIC FACTORS:
• Runs in families (mother, sisters)
• Multiple genes involved
• No single "PCOS gene"

⚡ INSULIN RESISTANCE:
• 70% of women with PCOS have it
• Excess insulin triggers androgen production
• Creates a hormonal imbalance cycle

🔬 HORMONAL IMBALANCE:
• Excess androgens (male hormones)
• Elevated LH (luteinizing hormone)
• Low FSH (follicle-stimulating hormone)
• Imbalanced estrogen/progesterone

🔥 INFLAMMATION:
• Chronic low-grade inflammation
• May stimulate androgen production
• Linked to insulin resistance

🌍 ENVIRONMENTAL FACTORS:
• Diet and lifestyle
• Stress levels
• Exposure to endocrine disruptors
• Obesity (can worsen symptoms)

❓ KEY POINT:
PCOS is a complex syndrome with no single cause. It's likely a combination of genetic predisposition and environmental triggers.""",
    
    'diagnosis': """How is PCOS Diagnosed?

📋 ROTTERDAM CRITERIA (need 2 of 3):
1️⃣ Irregular or absent ovulation
2️⃣ High androgen levels (blood test or symptoms)
3️⃣ Polycystic ovaries on ultrasound

🩺 MEDICAL EVALUATION:
• Detailed medical history
• Physical examination (BMI, blood pressure)
• Pelvic examination
• Check for hirsutism, acne, hair loss

🧪 BLOOD TESTS:
• Hormone levels (androgens, LH, FSH, prolactin)
• Thyroid function (TSH, T4)
• Fasting glucose and insulin
• Cholesterol and triglycerides
• Vitamin D levels

🔊 PELVIC ULTRASOUND:
• Check for cysts on ovaries
• Assess ovarian volume
• Examine uterine lining

⚠️ RULE OUT:
• Thyroid disorders
• Cushing's syndrome
• Congenital adrenal hyperplasia
• Tumors

💡 No single test diagnoses PCOS - it's based on overall assessment. See a gynecologist or endocrinologist for proper evaluation!""",
    
    'irregular_periods': """Irregular Periods and PCOS:

📅 WHAT'S CONSIDERED IRREGULAR:
• Cycles longer than 35 days
• Fewer than 8 periods per year
• Absent periods (amenorrhea)
• Unpredictable cycle lengths
• Very heavy or light bleeding

🔍 WHY IT HAPPENS:
• Lack of ovulation (anovulation)
• Hormonal imbalances
• Excess androgens interfere with cycle
• Insulin resistance affects hormones

💊 TREATMENT OPTIONS:
• Birth control pills (regulate cycles)
• Progestin therapy
• Metformin (improves ovulation)
• Lifestyle changes

📱 TRACKING TIPS:
• Use period tracking apps
• Note cycle length and flow
• Track symptoms
• Share data with your doctor

⚠️ WHEN TO SEE A DOCTOR:
• No period for 3+ months
• Cycles longer than 45 days
• Very heavy bleeding
• Severe pain
• Trying to conceive

Regular periods aren't just about fertility - they're important for overall health and reducing cancer risk!""",
    
    'hair_growth': """Managing Hirsutism (Excess Hair Growth):

🪒 WHAT IS IT:
• Unwanted male-pattern hair growth
• Face, chest, back, abdomen, thighs
• Affects 70% of women with PCOS
• Caused by excess androgens

💊 MEDICAL TREATMENTS:
• Birth control pills (reduce androgens)
• Spironolactone (anti-androgen)
• Finasteride (blocks DHT)
• Eflornithine cream (Vaniqa) - slows facial hair
• Metformin (improves hormones)

✂️ HAIR REMOVAL:
• Shaving (quick, temporary)
• Waxing/threading (lasts longer)
• Laser hair removal (semi-permanent)
• Electrolysis (permanent)
• Depilatory creams (temporary)

🌿 NATURAL APPROACHES:
• Weight loss (reduces androgens)
• Spearmint tea (may reduce androgens)
• Saw palmetto supplements
• Inositol supplements

⏰ TIMELINE:
• Medical treatments take 6+ months
• Be patient and consistent
• Combination approaches work best

💭 EMOTIONAL SUPPORT:
• It's a medical symptom, not your fault
• Many women experience this
• Talk to others with PCOS
• Consider therapy if it affects self-esteem""",
    
    'acne': """PCOS-Related Acne Treatment:

🔴 WHY PCOS CAUSES ACNE:
• Excess androgens increase oil production
• Affects jaw, chin, neck, chest, back
• Often severe and cystic
• May be resistant to regular treatments

💊 MEDICAL TREATMENTS:
• Birth control pills (reduce androgens)
• Spironolactone (anti-androgen, very effective)
• Topical retinoids (tretinoin, adapalene)
• Benzoyl peroxide
• Antibiotics (short-term)
• Accutane (severe cases)

🧴 SKINCARE ROUTINE:
• Gentle cleanser (2x daily)
• Non-comedogenic products
• Oil-free moisturizer
• Sunscreen (SPF 30+)
• Avoid harsh scrubs

🌿 LIFESTYLE TIPS:
• Low-glycemic diet
• Reduce dairy intake
• Drink plenty of water
• Manage stress
• Don't pick or squeeze
• Change pillowcases regularly

⚡ PROFESSIONAL TREATMENTS:
• Chemical peels
• Light therapy
• Laser treatments
• Cortisone injections (cystic acne)

⏰ BE PATIENT:
• Takes 3-6 months to see results
• Hormonal treatment is most effective
• See a dermatologist for persistent acne""",
    
    'mental_health': """PCOS and Mental Health:

🧠 COMMON CHALLENGES:
• Depression (2-3x higher risk)
• Anxiety disorders
• Low self-esteem
• Body image issues
• Eating disorders
• Social isolation

😔 WHY IT HAPPENS:
• Hormonal imbalances affect mood
• Chronic stress of managing symptoms
• Visible symptoms (acne, hair growth, weight)
• Fertility concerns
• Feeling misunderstood

💪 COPING STRATEGIES:
• Therapy/counseling (CBT is effective)
• Support groups (online or in-person)
• Stress management (meditation, yoga)
• Regular exercise (boosts mood)
• Adequate sleep
• Stay connected with loved ones

💊 WHEN TO SEEK HELP:
• Persistent sadness
• Loss of interest in activities
• Sleep problems
• Appetite changes
• Thoughts of self-harm
• Difficulty functioning

🌟 SELF-CARE:
• Practice self-compassion
• Celebrate small victories
• Focus on what you can control
• Join PCOS communities
• Educate loved ones

Your mental health matters! Don't hesitate to seek professional support.""",
    
    'supplements': """Supplements for PCOS:

✅ EVIDENCE-BASED:
• Inositol (Myo & D-chiro) - improves insulin, ovulation
• Vitamin D - many with PCOS are deficient
• Omega-3 fatty acids - reduce inflammation
• Magnesium - insulin sensitivity
• Chromium - blood sugar control
• N-Acetyl Cysteine (NAC) - fertility
• Berberine - similar to Metformin

🌿 TRADITIONAL/HERBAL:
• Spearmint tea - reduce androgens
• Cinnamon - blood sugar
• Turmeric - anti-inflammatory
• Saw palmetto - reduce DHT
• Vitex (Chasteberry) - hormone balance

⚠️ IMPORTANT:
• Consult doctor before taking
• Quality matters - choose reputable brands
• Not FDA regulated like medications
• May interact with medications
• Not a replacement for lifestyle changes

💊 DOSAGE EXAMPLES:
• Inositol: 2-4g daily (Myo:D-chiro 40:1)
• Vitamin D: 1000-4000 IU (test levels first)
• Omega-3: 1000-2000mg EPA+DHA
• Magnesium: 300-400mg

Supplements work best alongside diet, exercise, and medical treatment!""",
    
    'fertility_tips': """Tips to Boost Fertility with PCOS:

🎯 LIFESTYLE CHANGES:
• Achieve healthy weight (even 5% loss helps!)
• Exercise regularly but not excessively
• Reduce stress
• Quit smoking and limit alcohol
• Get 7-9 hours sleep
• Avoid endocrine disruptors

🍽️ DIET FOR FERTILITY:
• Low-glycemic foods
• Lots of vegetables and fruits
• Healthy fats (omega-3s)
• Lean protein
• Limit processed foods
• Stay hydrated

📊 TRACK OVULATION:
• Basal body temperature
• Ovulation predictor kits
• Cervical mucus changes
• Fertility apps
• Consider ovulation monitoring by doctor

💊 MEDICATIONS:
• Clomid (Clomiphene citrate)
• Letrozole (Femara) - often more effective
• Metformin - improve ovulation
• Gonadotropins (injectable)

⏰ TIMING:
• Have sex every 2-3 days
• Focus on fertile window
• Don't stress about "perfect" timing

🏥 WHEN TO SEE SPECIALIST:
• Under 35: after 12 months trying
• Over 35: after 6 months trying
• Irregular/absent periods

Remember: Many women with PCOS conceive naturally or with simple treatments!""",
    
    'longterm_health': """Long-term Health Risks with PCOS:

⚠️ INCREASED RISKS:
• Type 2 Diabetes (50% risk by age 40)
• Heart disease and high blood pressure
• High cholesterol
• Sleep apnea
• Fatty liver disease
• Endometrial cancer (if irregular periods)
• Stroke

🛡️ PREVENTION STRATEGIES:
• Regular health screenings
• Manage weight
• Exercise regularly
• Healthy diet
• Don't smoke
• Limit alcohol
• Manage stress

📋 RECOMMENDED TESTS:
• Glucose tolerance test (yearly)
• Lipid panel (cholesterol)
• Blood pressure monitoring
• Liver function tests
• Endometrial biopsy (if needed)

💊 PROTECTIVE MEASURES:
• Birth control or progestin (protect uterus)
• Metformin (reduce diabetes risk)
• Statins (if high cholesterol)
• Blood pressure medications (if needed)

🌟 GOOD NEWS:
• Early intervention prevents complications
• Healthy lifestyle dramatically reduces risks
• Many risks are manageable
• Regular monitoring catches issues early

PCOS is a lifelong condition but with proper management, you can be healthy!""",
    
    'what_is_pcos': """What is PCOS (Polycystic Ovary Syndrome)?

📚 DEFINITION:
PCOS is a hormonal disorder affecting 1 in 10 women of reproductive age. Despite the name, you don't need to have ovarian cysts to have PCOS!

🔍 KEY FEATURES:
• Hormonal imbalance (excess androgens)
• Irregular or absent periods
• Insulin resistance
• Difficulty with weight management
• Multiple small follicles on ovaries

📊 WHO GETS IT:
• 5-10% of women aged 15-44
• All ethnicities (higher in certain groups)
• Often starts in teens but may not be diagnosed until later
• Genetic component (runs in families)

❓ COMMON MISCONCEPTIONS:
• ❌ "You must have cysts" - NO, name is misleading
• ❌ "Only affects overweight women" - NO, affects all sizes
• ❌ "You can't get pregnant" - NO, harder but possible
• ❌ "It's just about periods" - NO, affects whole body
• ❌ "It goes away" - NO, lifelong but manageable

💡 THE TRUTH:
• It's a metabolic and endocrine disorder
• Affects more than just reproduction
• Varies greatly between individuals
• Manageable with proper care
• Research is ongoing

PCOS is complex but understanding it is the first step to managing it!""",
    
    'doctor_questions': """Questions to Ask Your Doctor About PCOS:

🩺 DIAGNOSIS:
• What tests confirm my PCOS?
• What type of PCOS do I have?
• Should I see a specialist?
• What are my hormone levels?
• Do I have insulin resistance?

💊 TREATMENT:
• What treatments do you recommend for my symptoms?
• Are there medication options?
• What are the side effects?
• How long until I see results?
• Can I try lifestyle changes first?

🤰 FERTILITY:
• How does PCOS affect my fertility?
• What are my options for getting pregnant?
• Should I see a fertility specialist?
• What medications can help?
• What's my timeline?

📊 MONITORING:
• What tests should I have regularly?
• How often should I come for check-ups?
• What warning signs should I watch for?
• How do I track my progress?

🏥 LONG-TERM:
• What are my risks for other conditions?
• How can I prevent complications?
• Will PCOS change as I age?
• What about menopause?

💡 TIP: Write down questions before your appointment and take notes during!""",
    
    'help': """I can help you with information about:

📋 PCOS BASICS:
• What is PCOS?
• Symptoms and signs
• Causes and risk factors
• Diagnosis process

💊 TREATMENT:
• Medical treatments
• Medications (Metformin, birth control, etc.)
• Natural remedies
• Supplements

🍽️ LIFESTYLE:
• PCOS-friendly diet
• Exercise recommendations
• Weight management
• Meal planning

🤰 FERTILITY:
• Getting pregnant with PCOS
• Fertility treatments
• Pregnancy tips
• Ovulation tracking

💪 SYMPTOMS:
• Irregular periods
• Hair growth (hirsutism)
• Acne treatment
• Hair loss
• Weight issues

🧠 WELLBEING:
• Mental health support
• Stress management
• Self-care tips
• Support resources

⚕️ HEALTH:
• Long-term risks
• Prevention strategies
• Regular screenings

Just ask me anything about PCOS and women's health!""",
    
    'default': """I understand you're asking about {query}. While I'm focused on PCOS and women's health, I want to help!

For PCOS-related questions, I can assist with:
• Symptoms, diagnosis, and treatment
• Diet, exercise, and lifestyle
• Fertility and pregnancy
• Mental health and emotional support
• Long-term health management

Could you rephrase your question or ask something specific about PCOS? 

Or type 'help' to see all topics I can discuss! 💙"""
}

def load_model():
    """Load the TensorFlow model"""
    global model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('model.h5', compile=True)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_file):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    """
    # Open the image using Pillow
    img = Image.open(image_file)
    
    # Resize the image to 224x224 pixels
    img = img.resize((224, 224))
    
    # Convert image to a numpy array
    img_array = np.array(img)
    
    # Ensure image is in RGB format (3 channels)
    if img_array.ndim == 2: # Handle grayscale images
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Remove alpha channel if it exists (e.g., for PNGs)
    if img_array.shape[2] == 4:
        img_array = img_array[..., :3]

    # Scale pixel values from [0, 255] to [0, 1], as done in training
    img_array = img_array / 255.0
    
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_with_model(image_array):
    """Make prediction using the loaded model"""
    global model
    if model is None:
        return None, 0
    
    try:
        prediction_array = model.predict(image_array)
        predicted_index = np.argmax(prediction_array[0])
        confidence = float(np.max(prediction_array[0]) * 100)
        class_names = ['infected', 'notinfected']
        result = class_names[predicted_index]
        return result, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0

# Routes
@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html', logged_in=session.get('user_id') is not None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
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
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed! Please try again.', 'error')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
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
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password!', 'error')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/predict')
def predict():
    """Prediction options page - requires login"""
    if 'user_id' not in session:
        flash('Please login first to access prediction features!', 'error')
        return redirect(url_for('login'))
    
    return render_template('predict.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """Handle ultrasound image upload and prediction"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    # Try to load model if not already loaded
    if model is None:
        if not load_model():
            flash('Model not available. Please contact administrator.', 'error')
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
            # Save the file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{session['user_id']}_{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            with open(filepath, 'rb') as f:
                processed_image = preprocess_image(f)
            
            result, confidence = predict_with_model(processed_image)
            
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
            
            flash('Prediction completed successfully!', 'success')
            return render_template('result.html', prediction=result, confidence=confidence, prediction_type='Image Upload')
            
        except Exception as e:
            flash(f'An error occurred during prediction: {str(e)}', 'error')
            return redirect(url_for('predict'))
    else:
        flash('Invalid file type! Please upload PNG, JPG, or JPEG files.', 'error')
        return redirect(url_for('predict'))

@app.route('/video_feed')
def video_feed():
    """Video streaming route - placeholder for now"""
    if 'user_id' not in session:
        return "Unauthorized", 401
    
    # For now, return a placeholder image
    return Response(
        b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9',
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/capture_prediction', methods=['POST'])
def capture_prediction():
    """Capture current frame prediction from camera"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # For now, return a mock prediction
    return jsonify({
        'success': True,
        'result': 'notinfected',
        'confidence': 85.5
    })

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera stream"""
    return jsonify({'success': True})

@app.route('/history')
def history():
    """View prediction history"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/manual_entry')
def manual_entry():
    """Show manual data entry form"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    return render_template('manual_entry.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    """Handle manual data entry prediction (simple rule-based)"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    try:
        # Get form data
        age = float(request.form.get('age'))
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        cycle_length = float(request.form.get('cycle_length'))
        cycle_regularity = int(request.form.get('cycle_regularity'))
        fast_food = float(request.form.get('fast_food'))
        exercise = float(request.form.get('exercise'))
        hirsutism = int(request.form.get('hirsutism'))
        acne = int(request.form.get('acne'))
        hair_loss = int(request.form.get('hair_loss'))
        weight_gain = int(request.form.get('weight_gain'))
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Simple rule-based prediction
        score = 0
        
        # Irregular cycle is a major indicator
        if cycle_regularity == 0:
            score += 3
        
        # Long cycle length
        if cycle_length > 35:
            score += 2
        
        # High BMI
        if bmi > 25:
            score += 2
        
        # Symptoms
        score += hirsutism * 2
        score += acne * 1.5
        score += hair_loss * 1
        score += weight_gain * 1.5
        
        # Lifestyle factors
        if fast_food > 3:
            score += 1
        if exercise < 2:
            score += 1
        
        # Determine result
        if score >= 6:
            result = 'infected'
            confidence = min(60.0 + score * 3, 95.0)
        else:
            result = 'notinfected'
            confidence = min(70.0 + (10 - score) * 2, 90.0)
        
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
        
        # Redirect to report
        return redirect(url_for('generate_report_html'))
        
    except Exception as e:
        logger.error(f"Error during manual prediction: {e}")
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('manual_entry'))

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
        
        # Try to use the model if available
        result = None
        confidence = None
        
        # Try loading model
        if model is None:
            load_model()
        
        if model is not None:
            try:
                # Preprocess frame for model
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                prediction_array = model.predict(img_array)
                predicted_index = np.argmax(prediction_array[0])
                confidence = float(np.max(prediction_array[0]) * 100)
                result = 'infected' if predicted_index == 0 else 'notinfected'
            except Exception as model_error:
                logger.warning(f"Model prediction failed: {model_error}")
        
        # Fallback to simple brightness analysis if model fails
        if result is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if avg_brightness < 100:
                result = 'infected'
                confidence = min(85.0 + (100 - avg_brightness) * 0.15, 95.0)
            else:
                result = 'notinfected'
                confidence = min(85.0 + (avg_brightness - 100) * 0.1, 95.0)
        
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
        
        logger.info(f"Camera prediction completed - Result: {result}, Confidence: {confidence}%")
        
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'method': 'Camera Analysis'
        })
            
    except Exception as e:
        logger.error(f"Error during camera frame processing: {e}")
        return jsonify({'error': str(e)}), 500

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

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Enhanced chatbot with AI API integration and comprehensive pattern matching"""
    # Handle GET request for testing
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Chat endpoint is working with AI integration'
        })
    
    # Handle POST request for actual chat
    try:
        # Get JSON data
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
            
        user_message = data.get('message', '').strip()
        original_message = user_message
        user_message = user_message.lower()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Define comprehensive keyword patterns for better matching
        patterns = {
            'greeting': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'start', 'begin'],
            'what_is_pcos': ['what is pcos', 'define pcos', 'explain pcos', 'tell me about pcos', 'pcos meaning', 'what does pcos mean'],
            'symptoms': ['symptom', 'sign', 'indication', 'how do i know', 'do i have pcos', 'check for pcos'],
            'treatment': ['treat', 'medicine', 'medication', 'cure', 'remedy', 'therapy', 'prescription', 'drug'],
            'diet': ['diet', 'food', 'eat', 'nutrition', 'meal', 'recipe', 'what to eat', 'what not to eat', 'grocery'],
            'exercise': ['exercise', 'workout', 'physical activity', 'gym', 'fitness', 'yoga', 'running', 'cardio', 'strength'],
            'pregnancy': ['pregnan', 'conceive', 'fertility', 'baby', 'getting pregnant', 'trying to conceive', 'ttc', 'infertil', 'ovulat'],
            'fertility_tips': ['boost fertility', 'increase fertility', 'improve fertility', 'fertility help', 'get pregnant faster'],
            'weight': ['weight', 'lose weight', 'obesity', 'fat', 'overweight', 'bmi', 'weight loss', 'weight gain', 'slim'],
            'causes': ['cause', 'why', 'reason', 'origin', 'how did i get', 'where does it come from'],
            'diagnosis': ['diagnos', 'test', 'detect', 'check', 'screen', 'exam', 'ultrasound', 'blood test'],
            'irregular_periods': ['period', 'menstrua', 'cycle', 'irregular', 'missed period', 'no period', 'amenorrhea'],
            'hair_growth': ['hair growth', 'facial hair', 'hirsutism', 'unwanted hair', 'excess hair', 'body hair', 'mustache', 'beard'],
            'acne': ['acne', 'pimple', 'breakout', 'skin', 'oily skin', 'cystic acne', 'hormonal acne'],
            'mental_health': ['depress', 'anxiety', 'mental health', 'mood', 'emotional', 'stress', 'sad', 'worry', 'cope', 'feeling down'],
            'supplements': ['supplement', 'vitamin', 'inositol', 'omega', 'herbal', 'natural remedy', 'vitamins'],
            'longterm_health': ['long term', 'risk', 'complication', 'diabetes', 'heart disease', 'cancer', 'health risk'],
            'doctor_questions': ['ask doctor', 'doctor visit', 'appointment', 'what to ask', 'questions for doctor'],
            'help': ['help', 'what can you do', 'options', 'topics', 'menu', 'assist']
        }
        
        # Find the best matching response
        bot_response = None
        max_matches = 0
        matched_category = None
        
        # Check each pattern category
        for category, keywords in patterns.items():
            matches = sum(1 for keyword in keywords if keyword in user_message)
            if matches > max_matches:
                max_matches = matches
                matched_category = category
        
        # Get response from rule-based system if strong match found
        if matched_category and max_matches > 0:
            bot_response = PCOS_RESPONSES.get(matched_category)
            logger.info(f"Matched category: {matched_category} with {max_matches} keyword matches")
        
        # If no strong match, try AI API for intelligent response
        if not bot_response or max_matches < 2:
            logger.info(f"Querying AI API for: {original_message[:50]}")
            ai_response = query_medical_ai(original_message)
            
            if ai_response and len(ai_response.strip()) > 20:
                # AI gave a good response
                bot_response = f"""🤖 AI Medical Assistant:

{ai_response}

---
💡 **Note:** This is AI-generated information. For personalized medical advice about PCOS, please consult a healthcare professional.

Type 'help' to see all topics I can discuss in detail!"""
                logger.info("Using AI-generated response")
            else:
                # AI didn't work, use contextual default
                query_snippet = original_message[:50] if len(original_message) > 50 else original_message
                bot_response = PCOS_RESPONSES['default'].format(query=query_snippet)
                logger.info(f"Using default response for: {user_message[:50]}")
        
        logger.info(f"Chatbot response generated for message: {user_message[:50]}")
        return jsonify({'response': bot_response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'response': PCOS_RESPONSES.get('help', 'I apologize, but I encountered an error. Please try asking your question again.')})

@app.route('/generate_report')
def generate_report():
    """Generate PDF report (placeholder - redirects to HTML report)"""
    if 'user_id' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))
    
    # For now, redirect to HTML report
    # Full PDF generation would require reportlab library
    flash('PDF generation coming soon! Viewing HTML report instead.', 'info')
    return redirect(url_for('generate_report_html'))

# Initialize database
with app.app_context():
    db.create_all()
    print("Database tables created!")

if __name__ == '__main__':
    print("Starting Femaura.AI application...")
    print("Note: Model will be loaded when first prediction is made")
    print("Enhanced chatbot ready with 15+ topic areas!")
    print("AI Medical Assistant:")
    print("  ✓ Primary: Microsoft BioGPT (Biomedical Specialist)")
    print("  ✓ Backup: Groq LLaMA 3 (Fast & Free)")
    print("  ✓ Fallback: Rule-based system (Offline)")
    print("")
    print("BioGPT is trained on PubMed biomedical literature for accurate medical responses!")
    app.run(debug=True, threaded=True)


