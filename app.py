import streamlit as st
import whisper
import tempfile
from deep_translator import GoogleTranslator
from symptoms_list import symptoms
from drug_list import drugs
from severity_rules import high_risk_symptoms, moderate_risk_symptoms, high_risk_drugs, moderate_risk_drugs

# Custom CSS for beautiful cards
card_style = """
<style>
.result-card {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    color: white;
    font-size: 18px;
}
.low-risk {
    background-color: #4CAF50; /* green */
}
.moderate-risk {
    background-color: #FF9800; /* orange */
}
.high-risk {
    background-color: #F44336; /* red */
}
.section-title {
    font-size: 22px;
    font-weight: bold;
    margin-top: 18px;
}
</style>
"""
st.markdown(card_style, unsafe_allow_html=True)



def extract_symptoms(text):
    text = text.lower()
    detected = []

    for s in symptoms:
        if s in text:
            detected.append(s)

    # Remove duplicates
    return list(set(detected))

def extract_drugs(text):
    text = text.lower()
    detected = []

    for d in drugs:
        if d in text:
            detected.append(d)

    return list(set(detected))

def calculate_risk(detected_symptoms, detected_drugs):
    score = 0

    # Score symptoms
    for s in detected_symptoms:
        if s in high_risk_symptoms:
            score += 3
        elif s in moderate_risk_symptoms:
            score += 1

    # Score drugs
    for d in detected_drugs:
        if d in high_risk_drugs:
            score += 3
        elif d in moderate_risk_drugs:
            score += 1

    # Final risk label
    if score >= 5:
        return "HIGH", score
    elif score >= 2:
        return "MODERATE", score
    else:
        return "LOW", score

def render_risk_card(risk_label, detected_symptoms, detected_drugs):
    if risk_label == "LOW":
        css_class = "low-risk"
    elif risk_label == "MODERATE":
        css_class = "moderate-risk"
    else:
        css_class = "high-risk"

    symptoms_text = ", ".join(detected_symptoms) if detected_symptoms else "None detected"
    drugs_text = ", ".join(detected_drugs) if detected_drugs else "None detected"

    card_html = f"""
    <div class="result-card {css_class}">
        <div class="section-title">⚠️ Risk Level: {risk_label}</div>
        <p><strong>🩺 Symptoms:</strong> {symptoms_text}</p>
        <p><strong>💊 Drugs:</strong> {drugs_text}</p>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

# Title
st.title("🎧 Medical Audio Risk Detector (Prototype)")
st.write("Upload an audio file, transcribe it, and translate it to English.")

# Load Whisper model (small for speed)
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# File uploader
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

if audio_file:
    st.audio(audio_file)

    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        temp_audio_path = tmp.name

    st.info("🔍 Transcribing audio...")

    # Transcribe with Whisper
    result = model.transcribe(temp_audio_path, language="hi")  # auto-detect + Hindi-friendly
    transcript = result["text"]

    # ---- SHOW ORIGINAL TRANSCRIPTION ----
    st.subheader("📝 Transcribed Text (Detected Language):")
    st.write(transcript)

    # ---- TRANSLATE TO ENGLISH ----
    st.info("🌐 Translating to English...")

    try:
        english_text = GoogleTranslator(source="auto", target="en").translate(transcript)
    except:
        english_text = transcript  # fallback if translation fails

    st.subheader("🇬🇧 English Translation:")
    st.write(english_text)

        # ---- SYMPTOM EXTRACTION ----
    st.info("🧠 Extracting symptoms from English text...")

    detected_symptoms = extract_symptoms(english_text)

    st.subheader("🩺 Detected Symptoms:")
    if detected_symptoms:
        st.write(", ".join(detected_symptoms))
    else:
        st.write("No recognizable symptoms found.")

    # ---- DRUG EXTRACTION ----
    st.info("💊 Detecting drugs mentioned in text...")

    detected_drugs = extract_drugs(english_text)

    st.subheader("💊 Detected Drugs:")
    if detected_drugs:
        st.write(", ".join(detected_drugs))
    else:
        st.write("No known drugs found.")

    st.info("🚨 Calculating risk severity...")

    risk_label, risk_score = calculate_risk(detected_symptoms, detected_drugs)

    # Show final card
    render_risk_card(risk_label, detected_symptoms, detected_drugs)


    # NEXT STEPS placeholder
    st.success("Audio processed successfully! Ready for symptom & drug extraction next.")

