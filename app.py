import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

#EXCEL_PATH = "C:\\Users\\Asus\\Downloads\\symptoms_cleaned.xlsx"
# file_path ="D:\AIBootcamp\Ayurclf\style.css"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(BASE_DIR, st.secrets["paths"]["CSS_PATH"])
EXCEL_PATH= os.path.join(BASE_DIR, st.secrets["paths"]["EXCEL_PATH"])
df = pd.read_excel(EXCEL_PATH)



def load_css(file_path: str):
    with open(file_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css(file_path)

# ----------------------------
# Load Hierarchical ML Models
# ----------------------------
model_data = joblib.load("category_hierarchical_models.pkl")
model_super = model_data["model_super"]                     # ML super category classifier        
super_label_encoder = model_data["super_label_encoder"]     # label encoder for super category    


# ----------------------------
# PART 1 → Rule-based Matching
# ----------------------------
def get_top_super_categories(input_symptoms, language=None, top_n=3, excel_path=EXCEL_PATH):
    df = pd.read_excel(excel_path)

    df['symptom_col'] = df[language].astype(str).str.strip().str.lower()
    df['super_col'] = df['super_category'].astype(str).str.strip().str.lower()

    input_symptoms = [s.strip().lower() for s in input_symptoms if str(s).strip() != ""]
    input_set = set(input_symptoms)

    super_to_symptoms = df.groupby('super_col')['symptom_col'].apply(lambda s: set(s.dropna())).to_dict()

    results = []
    for sc, symptoms in super_to_symptoms.items():
        matched = input_set.intersection(symptoms)

        if len(matched) == 0:
            continue

        prob = len(matched) / len(input_symptoms)
        results.append({
            "super_category": sc,
            "matched_count": len(matched),
            "probability": prob,
            "matched_symptoms": ", ".join(sorted(matched))
        })

    results = sorted(results, key=lambda r: (r["probability"], r["matched_count"]), reverse=True)
    return results[:top_n]


# ----------------------------
# PART 2 → ML-based Prediction
# ----------------------------
def predict_super_ML(symptoms_text):
    """
    ML-based prediction using classifier from joblib.
    """
    
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    emb = embedder.encode([symptoms_text])
    probs = model_super.predict_proba(emb)[0]

    top_idx = np.argsort(probs)[::-1][:3]

    results = []
    for i in top_idx:
        label = str(super_label_encoder.inverse_transform([i])[0])
        results.append({
            "super_category": label,
            "probability": probs[i]
        })

    return results


# ----------------------------
# donut chart
# ----------------------------
def donut(label, pct):
    fig = go.Figure(
        data=[go.Pie(
            labels=[label, ""],
            values=[pct, 1 - pct],
            hole=0.65,
            direction='clockwise',
            textinfo='none',
            hoverinfo='none'
        )]
    )

    fig.update_layout(
        width=230,        
        height=230,       
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=False,
        annotations=[
            dict(text=f"{label}<br>{pct*100:.1f}%", x=0.5, y=0.5,
                 font=dict(size=16), showarrow=False)
        ]
    )
    return fig



# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ayumitra:Ayurvedic Symptom Classifier", layout="wide")
#st.title("Ayumitra: Ayurvedic Symptom Classifier")

st.markdown("""<style> .stApp {background: #FFFF; color: #0f172a;} </style>""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<div class="font-size">🤖 Ayumitra </div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">This is an ayurvedic bot trained under professional ayurvedic doctors to provide ayurvedic category prediction.</div>', unsafe_allow_html=True)

    st.header("⚙️ Configuration")
    mode = st.radio("Choose Prediction Method", ["Part 1 - Rule Based", "Part 2 - ML Based"])

    language = st.selectbox(
        "Select language of symptoms:",
        ["sanskrit", "marathi", "hindi", "english"]
    )

    top_n = st.slider("Number of super categories:", 1, 5, 3)

st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">How can I assist you today?</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

symptoms = st.multiselect(
    "Select your symptoms:",
    df[language].dropna().unique().tolist(),
    accept_new_options=True,
    key="symptoms_input"
)

symptoms_text = ", ".join(symptoms)

if st.button("Predict Super Category"):
    if symptoms_text.strip() == "":
        st.error("Please select or enter at least one symptom.")
    else:
        #st.subheader("📊 Prediction Results")

        # ----- PART 1 -----
        if mode == "Part 1 - Rule Based":
            input_symptoms = [s.strip() for s in symptoms_text.split(",")]
            results = get_top_super_categories(input_symptoms, language, top_n)

        # ----- PART 2 -----
        else:
            results = predict_super_ML(symptoms_text)

        if not results:
            st.warning("No predictions could be generated.")
        else:
            for r in results:
                col1, col2 = st.columns([1.2, 2])

                # donut chart
                with col1:
                    st.plotly_chart(
                        donut(r["super_category"], float(r["probability"])),
                        use_container_width=True
                    )

                # info box
                with col2:
                    st.write(f"### {r['super_category'].upper()}")
                    if mode == "Part 1 - Rule Based":
                        st.write(f"**Matched Symptoms:** {r['matched_symptoms']}")
                        st.write(f"**Matched Count:** {r['matched_count']}")
                    st.write(f"**Probability:** {r['probability']*100:.2f}%")
                    st.write("---")

st.markdown("---")
st.markdown(
    '<div class="footer">'
    'Powered by VNIT(Nagpur) | Ayurvedic Knowledge Base'
    '</div>', 
    unsafe_allow_html=True
)



