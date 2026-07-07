# -------------------- Imports --------------------
import os
import uuid
import cv2
import sqlite3
import numpy as np
from flask import (
    Flask, render_template, request, redirect, session,
    flash, url_for, jsonify, Response
)
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------- Flask Config --------------------
app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Database Setup --------------------
DB_NAME = "users.db"

def init_db():
    """Initialize SQLite database for users."""
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            password TEXT NOT NULL
        )
        """)
        conn.commit()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

init_db()

# -------------------- Model Setup --------------------
img_size = (224, 224)
model_path = "best_mobilenetv2_model.h5"  # Path to your trained model
model = load_model(model_path)
print("✅ Model loaded successfully.")

# -------------------- Webcam Feed Generator --------------------
def gen_frames():
    """Generator that captures frames from webcam for live video feed."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("⚠️ Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Flask route to stream webcam video."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- Plant Classes --------------------
valid_classes = [
    "Aloe_vera", "Arjuna", "Ashwagandha", "Basil", "Bay_leaf",
    "Calendula", "Chamomile", "Cinnamon", "Elderberry", "Eucalyptus",
    "Fennel", "Garlic", "Ginkgo_biloba", "Gotu_Kola", "Hibiscus",
    "Lavender", "Moringa", "Neem", "Rosemary", "Sage",
    "Shankhpushpi", "St_Johns_Wort", "Tulsi", "Turmeric", "Valerian"
]

# -------------------- Medicinal Uses Dictionary --------------------
medicinal_uses = {
    "Aloe_vera": """Heals burns and cuts – Aloe vera gel soothes and accelerates healing of minor burns, wounds, and sunburns.
Reduces skin inflammation – Helps treat eczema, psoriasis, and rashes.
Acts as a natural moisturizer – Hydrates skin without clogging pores.
Improves digestive health – Aloe vera juice can relieve constipation.
Supports wound healing – Prevents infection and promotes faster tissue repair.
Controls blood sugar – May help lower glucose and improve insulin sensitivity.
Boosts immunity – Detoxifies and strengthens the immune system.
Promotes hair growth – Reduces dandruff and nourishes the scalp.
Soothes oral problems – Reduces gum inflammation and plaque buildup.
Detoxifies the body – Helps flush out toxins and maintain wellness.""",

    "Arjuna": """Supports heart health – Strengthens heart muscles and improves cardiac function.
Controls blood pressure – Regulates high BP and improves circulation.
Reduces cholesterol – Lowers LDL (bad cholesterol).
Acts as an antioxidant – Protects heart tissues from oxidative stress.
Improves angina – Relieves chest pain and improves blood flow.
Aids in wound healing – Anti-inflammatory and antibacterial.
Strengthens bones – Supports bone health.
Enhances respiratory health – Useful for asthma and cough.
Supports liver health – Detoxifies and protects the liver.
Boosts stamina – Improves energy and reduces cardiac stress.""",

    "Ashwagandha": """Reduces stress – Lowers cortisol and promotes calmness.
Boosts energy – Improves endurance and reduces fatigue.
Enhances brain function – Improves memory and focus.
Improves sleep – Treats insomnia naturally.
Supports immunity – Strengthens defense mechanisms.
Balances hormones – Regulates thyroid and reproductive health.
Supports heart health – Reduces BP and cholesterol.
Increases muscle strength – Enhances recovery post-exercise.
Stabilizes blood sugar – Controls glucose levels.
Slows aging – Rich in antioxidants for longevity.""",

    "Basil": """Boosts immunity – Strengthens immune response.
Relieves respiratory problems – Effective for asthma and cough.
Reduces stress – Natural adaptogen for calmness.
Supports heart health – Regulates cholesterol and BP.
Improves digestion – Reduces bloating and acidity.
Fights inflammation – Contains eugenol with anti-inflammatory effect.
Controls blood sugar – Maintains glucose balance.
Protects against infections – Antibacterial and antifungal.
Promotes oral health – Prevents gum disease.
Rich in antioxidants – Slows aging.""",

    "Bay_leaf": """Supports digestion – Relieves indigestion and bloating.
Manages diabetes – Improves insulin function.
Boosts immunity – Rich in vitamin C.
Improves heart health – Reduces cholesterol.
Relieves cold – Steam inhalation eases congestion.
Fights inflammation – Reduces pain and swelling.
Aids wound healing – Prevents infection.
Promotes relaxation – Calms stress and improves sleep.
Supports kidneys – Acts as mild diuretic.
Improves skin – Reduces acne and irritation.""",

    "Calendula": """Heals wounds – Speeds healing of cuts and burns.
Soothes irritation – Eases eczema and rashes.
Fights infections – Antibacterial and antifungal.
Improves oral health – Reduces gum inflammation.
Relieves diaper rash – Protects baby’s skin.
Supports digestion – Eases gastritis and ulcers.
Boosts immunity – Strengthens body defenses.
Reduces menstrual cramps – Regulates hormones.
Calms sunburn – Repairs sun damage.
Promotes healthy skin – Reduces acne and scars.""",

    "Chamomile": """Promotes relaxation – Reduces stress and anxiety.
Improves sleep – Gentle sedative properties.
Soothes digestion – Relieves gas and cramps.
Reduces inflammation – Calms tissues.
Heals skin – Treats rashes and burns.
Supports oral health – Eases mouth sores.
Relieves cramps – Eases menstrual pain.
Boosts immunity – Fights infections.
Eases cold symptoms – Clears nasal congestion.
Protects scalp – Reduces dandruff.""",

    "Cinnamon": """Regulates blood sugar – Improves insulin sensitivity.
Supports heart health – Lowers LDL cholesterol.
Fights infections – Antibacterial and antifungal.
Reduces inflammation – Cinnamaldehyde compound.
Improves digestion – Eases bloating.
Boosts immunity – Antioxidant rich.
Relieves cough – Soothes throat.
Enhances brain function – Supports memory.
Promotes oral health – Prevents gum disease.
Natural preservative – Prevents food spoilage.""",

    "Elderberry": """Boosts immunity – High in antioxidants and vitamin C.
Fights flu – Shortens cold duration.
Reduces inflammation – Contains anthocyanins.
Supports lungs – Eases sore throat and cough.
Improves heart health – Reduces BP.
Promotes healthy skin – Fights free radicals.
Supports digestion – Mild laxative.
Fights infections – Antiviral properties.
Aids weight control – Low calorie, nutrient rich.
Protects against oxidative stress – Neutralizes free radicals.""",

    "Eucalyptus": """Clears congestion – Steam inhalation relieves sinus.
Soothes sore throat – Used in lozenges.
Decongestant – Cineole opens airways.
Relieves joint pain – Reduces arthritis pain.
Fights infections – Antibacterial.
Promotes oral health – Prevents plaque.
Reduces inflammation – Calms skin.
Repels insects – Natural repellent.
Boosts focus – Improves alertness.
Heals wounds – Prevents infection.""",

    "Fennel": """Improves digestion – Eases bloating and constipation.
Relieves menstrual cramps – Regulates cycle.
Boosts lactation – For breastfeeding mothers.
Supports respiratory health – Clears mucus.
Reduces inflammation – Antioxidant rich.
Controls BP – Potassium balance.
Freshens breath – Oral antibacterial.
Supports weight management – Appetite control.
Enhances metabolism – Improves absorption.
Protects against oxidative stress – Flavonoid rich.""",

    "Garlic": """Boosts immunity – Prevents infections.
Reduces BP – Lowers hypertension.
Improves heart health – Lowers cholesterol.
Natural antibiotic – Kills microbes.
Supports lungs – Relieves cough.
Controls sugar – Manages diabetes.
Reduces inflammation – Fights oxidative stress.
Detoxifies liver – Removes toxins.
Supports bone health – Strengthens bones.
Enhances endurance – Reduces fatigue.""",

    "Ginkgo_biloba": """Improves memory – Enhances focus and learning.
Supports brain – Increases blood flow.
Reduces anxiety – Calms nerves.
Supports eyes – Prevents degeneration.
Improves circulation – Strengthens vessels.
Reduces dementia symptoms – Supports brain.
Antioxidant – Prevents free radical damage.
Heart health – Improves vessels.
Relieves tinnitus – Eases ringing in ears.
Supports lungs – Improves oxygenation.""",

    "Gotu_Kola": """Heals wounds – Boosts collagen.
Enhances memory – Improves focus.
Reduces stress – Natural adaptogen.
Supports veins – Improves circulation.
Improves skin – Reduces stretch marks.
Supports liver – Detoxifies.
Aids digestion – Heals ulcers.
Reduces inflammation – For arthritis.
Supports breathing – Relieves asthma.
Boosts immunity – Strengthens defense.""",

    "Hibiscus": """Lowers BP – Manages hypertension.
Supports heart – Reduces cholesterol.
Boosts immunity – Vitamin C rich.
Aids digestion – Prevents constipation.
Liver health – Detoxifies.
Reduces inflammation – Soothes tissues.
Weight loss – Reduces fat.
Skin health – Prevents acne.
Relieves cramps – Regulates periods.
Diuretic – Removes toxins.""",

    "Lavender": """Reduces stress – Calms the mind.
Improves sleep – Treats insomnia.
Relieves headaches – Soothes migraines.
Heals skin – Treats burns and acne.
Eases muscle pain – Relieves soreness.
Boosts breathing – Clears sinuses.
Improves mood – Reduces depression.
Heals wounds – Prevents infection.
Promotes hair growth – Reduces dandruff.
Repels insects – Keeps mosquitoes away.""",

    "Moringa": """Nutrient rich – Vitamins and minerals.
Boosts immunity – Strengthens system.
Reduces inflammation – Eases arthritis.
Heart health – Lowers cholesterol.
Controls sugar – Manages diabetes.
Aids digestion – Reduces bloating.
Improves skin – Fights aging.
Protects liver – Detoxifies.
Boosts energy – Reduces fatigue.
Improves memory – Enhances cognition.""",

    "Neem": """Treats skin – Acne and eczema.
Antimicrobial – Fights bacteria and fungi.
Boosts immunity – Prevents infection.
Oral health – Prevents gum disease.
Regulates sugar – Manages diabetes.
Anti-inflammatory – Reduces pain.
Detoxifies – Cleanses blood.
Protects liver – Heals damage.
Insect repellent – Prevents bites.
Hair health – Strengthens roots.""",

    "Rosemary": """Improves memory – Boosts alertness.
Boosts immunity – Fights infections.
Aids digestion – Relieves bloating.
Reduces inflammation – Soothes joints.
Hair health – Stimulates growth.
Supports heart – Improves circulation.
Relieves cough – Clears chest.
Antimicrobial – Kills germs.
Heals skin – Reduces acne.
Boosts mood – Reduces stress.""",

    "Sage": """Boosts memory – Enhances cognition.
Reduces inflammation – Soothes joints.
Aids digestion – Reduces cramps.
Kills microbes – Antibacterial.
Soothes throat – Eases soreness.
Regulates sugar – Manages diabetes.
Improves oral health – Prevents gum issues.
Boosts immunity – Antioxidant rich.
Heals skin – Treats wounds.
Eases menopause – Balances hormones.""",

    "Shankhpushpi": """Improves memory – Boosts learning.
Reduces anxiety – Calms mind.
Improves sleep – Treats insomnia.
Protects brain – Prevents damage.
Boosts focus – Improves clarity.
Mild sedative – Relaxes nerves.
Improves mood – Eases depression.
Protects neurons – Antioxidant rich.
Improves alertness – Reduces fatigue.
Nervous health – Strengthens nerves.""",

    "St_Johns_Wort": """Fights depression – Improves mood.
Reduces anxiety – Calms nerves.
Improves sleep – Reduces insomnia.
Heals wounds – Speeds recovery.
Reduces inflammation – Soothes tissues.
Eases nerve pain – Relieves sciatica.
Balances mood – Regulates hormones.
Antioxidant – Protects cells.
Heals skin – Reduces rashes.
Helps SAD – Treats seasonal depression.""",

    "Tulsi": """Boosts immunity – Prevents infection.
Reduces stress – Calms mind.
Improves breathing – Clears congestion.
Regulates sugar – Manages diabetes.
Heart health – Lowers cholesterol.
Aids digestion – Reduces acidity.
Anti-inflammatory – Fights swelling.
Liver health – Detoxifies.
Improves skin – Prevents acne.
Improves focus – Boosts clarity.""",

    "Turmeric": """Anti-inflammatory – Treats arthritis.
Antioxidant – Prevents cell damage.
Heart health – Lowers cholesterol.
Boosts immunity – Fights infections.
Liver health – Detoxifies.
Aids digestion – Reduces bloating.
Improves skin – Treats acne.
Cancer prevention – Stops tumor growth.
Improves mood – Treats depression.
Protects brain – Prevents Alzheimer’s.""",

    "Valerian": """Improves sleep – Treats insomnia.
Reduces anxiety – Natural sedative.
Relaxes muscles – Reduces cramps.
Improves focus – Calms mind.
Relieves headache – Soothes tension.
Eases spasms – Smooth muscle relaxant.
Heart health – Lowers BP.
Eases menopause – Reduces hot flashes.
Pain relief – Reduces aches.
Nerve health – Calms system."""
}

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                (name, email, phone, hashed_password)
            )
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            flash("Login successful!", "success")
            return redirect(url_for("predict_plant"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/about")
def about():
    return render_template("about.html")

# -------------------- Prediction --------------------
@app.route("/predict", methods=["GET", "POST"])
def predict_plant():
    if "user_id" not in session:
        flash("Please login to access plant prediction.", "warning")
        return redirect(url_for("login"))

    prediction = None
    confidence = None
    uses_text = None
    filename = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            preds = model.predict(img_array)
            predicted_index = np.argmax(preds, axis=1)[0]
            prediction = valid_classes[predicted_index]
            confidence = round(preds[0][predicted_index] * 100, 2)
            uses_text = medicinal_uses.get(prediction, "No medicinal uses available.")

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        uses_text=uses_text,
        filename=filename,
    )

@app.route("/get_use/<plant_name>")
def get_use(plant_name):
    """Return medicinal uses of a plant as JSON."""
    use = medicinal_uses.get(plant_name, "No data available for this plant.")
    return jsonify({"plant": plant_name, "uses": use})

# -------------------- Run Flask App --------------------
if __name__ == "__main__":
    print(app.url_map)  # Shows all registered routes
    app.run(debug=True)
