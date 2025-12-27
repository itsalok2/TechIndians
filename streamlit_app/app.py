import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import json
import pandas as pd
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import xgboost as xgb

# ================================
# Load Model + Metadata
# ================================
@st.cache_resource
def load_assets():
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_severity_model.json")

    with open("models/label_encoders_mapping.json") as f:
        encoders = json.load(f)

    with open("models/model_features.json") as f:
        feature_order = json.load(f)

    # Create reverse mappings for all encoders
    prod_ai_reverse = {v: int(k) for k, v in encoders["prod_ai"].items()}
    indi_pt_reverse = {v: int(k) for k, v in encoders["indi_pt"].items()}
    pt_reverse = {v: int(k) for k, v in encoders["pt"].items()}
    role_cod_reverse = {v: int(k) for k, v in encoders["role_cod"].items()}

    return model, encoders, feature_order, prod_ai_reverse, indi_pt_reverse, pt_reverse, role_cod_reverse


model, encoders, FEATURE_ORDER, PROD_AI_MAP, INDI_PT_MAP, PT_MAP, ROLE_COD_MAP = load_assets()

# ================================
# Load Whisper
# ================================
@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

whisper_model = load_whisper()

# ================================
# Gemini LLM
# ================================

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Fixed: Escaped curly braces in JSON example with double braces
prompt = PromptTemplate(
    template="""
You are a medical information extractor.

Extract ONLY items present in BOTH:
1. The conversation
2. The provided lists

Return strict JSON:
{{
  "drugs": [],
  "indications": [],
  "reactions": []
}}

Conversation:
{conversation}

Valid Drugs:
{drug_list}

Valid Indications:
{indi_list}

Valid Reactions:
{pt_list}
""",
    input_variables=["conversation", "drug_list", "indi_list", "pt_list"]
)


parser = JsonOutputParser()

# ================================
# Entity Extraction
# ================================
def extract_entities(text):
    chain = prompt | llm | parser
    entities = chain.invoke({
        "conversation": text,
        "drug_list": list(PROD_AI_MAP.keys()),
        "indi_list": list(INDI_PT_MAP.keys()),
        "pt_list": list(PT_MAP.keys())
    })
    return entities

# ================================
# Feature Engineering
# ================================
def build_features(entities):
    rows = []
    drug_stats = encoders["drug_stats"]

    for drug in entities["drugs"]:
        if drug not in PROD_AI_MAP:
            continue

        drug_id = str(PROD_AI_MAP[drug])
        stats = drug_stats.get(drug_id, {})
        
        # Get first indication if available, else default to 0
        indi_encoded = 0
        if entities.get("indications") and len(entities["indications"]) > 0:
            first_indication = entities["indications"][0]
            if first_indication in INDI_PT_MAP:
                indi_encoded = INDI_PT_MAP[first_indication]
        
        # role_cod: PS=2 (Primary Suspect) is the default
        role_cod_encoded = 2  # 'PS' = Primary Suspect

        rows.append({
            "prod_ai_encoded": int(drug_id),
            "indi_pt_encoded": indi_encoded,
            "role_cod_encoded": role_cod_encoded,
            "drug_frequency": stats.get("drug_frequency", 0),
            "drug_avg_severity": stats.get("drug_avg_severity", 0.0),
            "num_drugs": len(entities["drugs"]),
            "num_reactions": len(entities.get("reactions", [])),
            "treatment_duration": 0,  # Default - could be extracted from text
            "dechal_binary": 0,
            "is_primary_suspect": 1,
            "is_secondary_suspect": 0,
            "is_concomitant": 0
        })

    return rows

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Medical Risk Detector", layout="wide")

st.title("🎧 Medical Audio Risk Detector")

st.markdown("""
Upload an audio file containing a medical conversation. The app will:
1. Transcribe the audio
2. Extract medical entities (drugs, indications, reactions)
3. Predict adverse event severity
""")

audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    st.audio(audio_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    try:
        st.info("🔍 Transcribing audio...")
        segments, info = whisper_model.transcribe(audio_path)
        transcript = " ".join([seg.text for seg in segments])

        st.subheader("📝 Transcription")
        st.write(transcript)

        st.info("🧠 Extracting medical entities...")
        entities = extract_entities(transcript)
        
        st.subheader("🔍 Extracted Entities")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Drugs:**")
            st.write(entities.get("drugs", []))
        with col2:
            st.write("**Indications:**")
            st.write(entities.get("indications", []))
        with col3:
            st.write("**Reactions:**")
            st.write(entities.get("reactions", []))

        st.info("⚙️ Building features...")
        feature_rows = build_features(entities)

        if not feature_rows:
            st.error("❌ No known drugs detected from the valid drug list.")
            st.stop()

        X = pd.DataFrame(feature_rows)[FEATURE_ORDER]
        
        st.subheader("📋 Feature Matrix")
        st.dataframe(X)

        st.info("🤖 Predicting severity...")
        preds = model.predict(X)
        proba = model.predict_proba(X)

        st.success("✅ Prediction complete")
        
        # Display results
        st.subheader("📊 Severity Predictions")
        
        for idx, (drug, pred, prob) in enumerate(zip(entities["drugs"], preds, proba)):
            severity_map = {0: "Low", 1: "Medium", 2: "High"}
            severity = severity_map.get(pred, "Unknown")
            confidence = prob[pred] * 100
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**Drug:** {drug}")
            with col2:
                st.write(f"**Severity:** {severity}")
            with col3:
                st.write(f"**Confidence:** {confidence:.1f}%")
            
            # Show probability distribution
            st.write("Probability distribution:")
            prob_df = pd.DataFrame({
                'Severity': ['Low', 'Medium', 'High'],
                'Probability': prob[idx]
            })
            st.bar_chart(prob_df.set_index('Severity'))
            st.divider()
    
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            os.unlink(audio_path)