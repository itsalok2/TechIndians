import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import json
import pandas as pd
import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import xgboost as xgb

# ================================
# Configuration
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# ================================
# Load Model + Metadata
# ================================
@st.cache_resource
def load_assets():
    """Load XGBoost model and all encoder mappings"""
    try:
        model = xgb.XGBClassifier()
        model_path = os.path.join(MODELS_DIR, "xgb_severity_model.json")
        model.load_model(model_path)

        encoders_path = os.path.join(MODELS_DIR, "label_encoders_mapping.json")
        with open(encoders_path) as f:
            encoders = json.load(f)

        features_path = os.path.join(MODELS_DIR, "model_features.json")
        with open(features_path) as f:
            feature_order = json.load(f)

        # Create reverse mappings for all encoders
        prod_ai_reverse = {v: int(k) for k, v in encoders["prod_ai"].items()}
        indi_pt_reverse = {v: int(k) for k, v in encoders["indi_pt"].items()}
        pt_reverse = {v: int(k) for k, v in encoders["pt"].items()}
        role_cod_reverse = {v: int(k) for k, v in encoders["role_cod"].items()}

        return model, encoders, feature_order, prod_ai_reverse, indi_pt_reverse, pt_reverse, role_cod_reverse
    
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error(f"Models directory: {MODELS_DIR}")
        st.error(f"Files in directory: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'Directory not found'}")
        sys.exit(1)


model, encoders, FEATURE_ORDER, PROD_AI_MAP, INDI_PT_MAP, PT_MAP, ROLE_COD_MAP = load_assets()

# ================================
# Load Whisper
# ================================
@st.cache_resource
def load_whisper():
    """Load Whisper model for audio transcription"""
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as e:
        st.error(f"❌ Error loading Whisper model: {str(e)}")
        sys.exit(1)

whisper_model = load_whisper()

# ================================
# Gemini LLM
# ================================
from dotenv import load_dotenv
load_dotenv()

# Verify API key is loaded
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in environment variables!")
    st.info("Please set GEMINI_API_KEY in your .env file")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# Fixed: Escaped curly braces in JSON example with double braces
prompt = PromptTemplate(
    template="""
You are a medical information extractor.
You'll be getting a transcribed text from a audio file and your task is to get the drugs which are present in the text 
how many drugs are there and from the conversation you have to find the indication point and the reactions,
These drugs indication point and reaction are the columns of my data on which my model is trained so I have to make inference 
Based on these points which you give to me.

I am giving you the list of drugs which I am using and the reaction and the 
indication point on which my model is trained you have to infer from the text file and you have to find out if they exist in the 
given list or not I am giving you the list

drug list: [

"ACETAMINOPHEN","ADALIMUMAB","CARBIDOPA\\LEVODOPA","DEXAMETHASONE","EFAVIRENZ\\EMTRICITABINE\\TENOFOVIR DISOPROXIL FUMARATE",
"EMTRICITABINE\\TENOFOVIR DISOPROXIL FUMARATE","ESOMEPRAZOLE MAGNESIUM","FUROSEMIDE","IBRUTINIB","INFLIXIMAB-DYYB","LANSOPRAZOLE",
"LENALIDOMIDE","MACITENTAN","NIVOLUMAB","OCTREOTIDE ACETATE","OMALIZUMAB","OMEPRAZOLE MAGNESIUM","PREDNISOLONE","PREDNISONE","RANITIDINE",
"RANITIDINE HYDROCHLORIDE","RIBOCICLIB","RITUXIMAB","RIVAROXABAN","RUXOLITINIB","SECUKINUMAB","TENOFOVIR DISOPROXIL FUMARATE","TOCILIZUMAB",
"TOFACITINIB CITRATE","VEDOLIZUMAB"

]

indication point list: [

"Abdominal discomfort","Acromegaly","Ankylosing spondylitis","Asthma","Atrial fibrillation","Breast cancer","Breast cancer metastatic",
"Carcinoid tumour","Chronic lymphocytic leukaemia","Chronic spontaneous urticaria","Colitis ulcerative","Crohn's disease",
"Diffuse large B-cell lymphoma","Dyspepsia","Gastric ulcer","Gastrooesophageal reflux disease","HIV infection","Malignant melanoma",
"Myelofibrosis","Neuroendocrine tumour", "Pain","Parkinson's disease","Plasma cell myeloma","Polycythaemia vera","Premedication",
"Prophylaxis","Psoriasis","Psoriatic arthropathy","Pulmonary arterial hypertension","Rheumatoid arthritis"

]

reaction: [

"Acute kidney injury","Anxiety","Arthralgia","Bladder cancer","Bone density decreased","Bone loss","Breast cancer","Chronic kidney disease",
"Colorectal cancer","Diarrhoea","Dyspnoea","End stage renal disease","Fatigue","Gastric cancer","Hepatic cancer","Lung neoplasm malignant",
"Multiple fractures","Nausea","Neoplasm malignant","Oesophageal carcinoma","Osteonecrosis","Osteoporosis","Pain","Pancreatic carcinoma",
"Pneumonia","Prostate cancer","Renal cancer","Renal failure","Renal injury","Skeletal injury"
]

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
    """Extract medical entities from text using Gemini LLM"""
    try:
        chain = prompt | llm | parser
        entities = chain.invoke({
            "conversation": text,
            "drug_list": list(PROD_AI_MAP.keys()),
            "indi_list": list(INDI_PT_MAP.keys()),
            "pt_list": list(PT_MAP.keys())
        })
        return entities
    except Exception as e:
        st.error(f"❌ Error extracting entities: {str(e)}")
        return {"drugs": [], "indications": [], "reactions": []}

# ================================
# Feature Engineering
# ================================
def build_features(entities):
    """Build feature matrix from extracted entities"""
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
1. 🎤 Transcribe the audio using Whisper
2. 🧠 Extract medical entities (drugs, indications, reactions) using Gemini AI
3. 📊 Predict adverse event severity using XGBoost
""")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ Information")
    st.write(f"**Supported Drugs:** {len(PROD_AI_MAP)}")
    st.write(f"**Supported Indications:** {len(INDI_PT_MAP)}")
    st.write(f"**Supported Reactions:** {len(PT_MAP)}")
    st.write(f"**Model Features:** {len(FEATURE_ORDER)}")
    
    st.divider()
    
    st.header("📁 Sample Data")
    if os.path.exists(DATA_DIR):
        audio_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.mp3', '.wav', '.m4a'))]
        if audio_files:
            st.write("Available audio files in data folder:")
            for audio in audio_files[:5]:  # Show first 5
                st.text(f"• {audio}")
        else:
            st.info("No audio files found in data folder")
    
    st.divider()
    st.caption("Powered by Whisper, Gemini AI & XGBoost")

# Main content
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    st.audio(audio_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    try:
        # Transcription
        with st.spinner("🔍 Transcribing audio..."):
            segments, info = whisper_model.transcribe(audio_path)
            transcript = " ".join([seg.text for seg in segments])

        st.subheader("📝 Transcription")
        with st.expander("View full transcript", expanded=True):
            st.write(transcript)

        # Entity Extraction
        with st.spinner("🧠 Extracting medical entities..."):
            entities = extract_entities(transcript)
        
        st.subheader("🔍 Extracted Entities")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Drugs Found", len(entities.get("drugs", [])))
            if entities.get("drugs"):
                for drug in entities["drugs"]:
                    st.success(f"💊 {drug}")
        with col2:
            st.metric("Indications Found", len(entities.get("indications", [])))
            if entities.get("indications"):
                for indi in entities["indications"]:
                    st.info(f"🏥 {indi}")
        with col3:
            st.metric("Reactions Found", len(entities.get("reactions", [])))
            if entities.get("reactions"):
                for react in entities["reactions"]:
                    st.warning(f"⚠️ {react}")

        # Feature Building
        with st.spinner("⚙️ Building features..."):
            feature_rows = build_features(entities)

        if not feature_rows:
            st.error("❌ No known drugs detected from the valid drug list.")
            st.info("The extracted drugs must match the drugs in the trained model.")
            st.stop()

        X = pd.DataFrame(feature_rows)[FEATURE_ORDER]
        
        with st.expander("📋 View Feature Matrix"):
            st.dataframe(X, use_container_width=True)

        # Prediction
        with st.spinner("🤖 Predicting severity..."):
            preds = model.predict(X)
            proba = model.predict_proba(X)

        st.success("✅ Prediction Complete!")
        
        # Display results
        st.subheader("📊 Severity Predictions")
        
        severity_map = {0: "Low", 1: "Medium", 2: "High"}
        severity_colors = {0: "🟢", 1: "🟡", 2: "🔴"}
        
        for idx, (drug, pred, prob) in enumerate(zip(entities["drugs"], preds, proba)):
            severity = severity_map.get(pred, "Unknown")
            severity_icon = severity_colors.get(pred, "⚪")
            confidence = prob[pred] * 100
            
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {severity_icon} {drug}")
                st.metric("Predicted Severity", severity)
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col2:
                st.write("**Probability Distribution:**")
                prob_df = pd.DataFrame({
                    'Severity': ['Low', 'Medium', 'High'],
                    'Probability': [prob[0], prob[1], prob[2]]
                })
                st.bar_chart(prob_df.set_index('Severity'), height=200)
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        import traceback
        with st.expander("View error details"):
            st.code(traceback.format_exc())
    
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            os.unlink(audio_path)

# Footer
st.divider()
st.caption("⚠️ This tool is for research purposes only. Always consult healthcare professionals for medical decisions.")