import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from utils import preprocess_image, load_class_names
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 🧠 Cache the model for performance
# ----------------------------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("model/plant_disease_model.h5")

# ----------------------------------------------------
# 🌿 Streamlit Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🍃",
    layout="centered",
)

# ----------------------------------------------------
# 🌾 Sidebar
# ----------------------------------------------------
st.sidebar.header("🌱 App Navigation")
st.markdown("""
<div style="
    text-align:center;
    font-size:20px;
    padding:12px;
    background-color:#121212;
    border-radius:10px;
    color:#ffffff;
    border:1px solid #2e7d32;
">
🌱 <b>AI-powered Plant Health Detection</b> | Built with ❤️ by <b style='color:#80ff80;'>Mukul Rajput</b>
</div>
""", unsafe_allow_html=True)


if st.sidebar.button("📚 View All Classes"):
    try:
        class_names = load_class_names()
        st.sidebar.success("Detected Classes:")
        for idx, cls in class_names.items():
            st.sidebar.write(f"• {cls}")
    except:
        st.sidebar.error("⚠️ Please train the model first!")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Developed by **Mukul Rajput**")
st.sidebar.markdown("⚙️ Powered by **TensorFlow & Streamlit**")

# ----------------------------------------------------
# 🌿 Main Content
# ----------------------------------------------------
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a **leaf image** and the model will predict the disease with confidence and suggestions.")

uploaded_file = st.file_uploader("📸 Upload Leaf Image", type=["jpg", "jpeg", "png"])

# ----------------------------------------------------
# 💊 Full Disease Information for 15 Classes
# ----------------------------------------------------
disease_info = {
    # -------- PEPPER --------
    "Pepper__bell___Bacterial_spot": {
        "type": "Infected",
        "description": "Caused by *Xanthomonas campestris*. Leads to small, dark, water-soaked lesions on leaves and fruits.",
        "prescription": [
            "⚠️ Remove and destroy infected plants.",
            "🧴 Apply copper-based bactericides.",
            "🚜 Avoid working with wet plants to reduce spread.",
            "🌞 Provide proper sunlight and ventilation."
        ]
    },
    "Pepper__bell___healthy": {
        "type": "Healthy",
        "description": "Your bell pepper plant appears healthy and thriving.",
        "prescription": [
            "✅ Maintain consistent watering.",
            "🌿 Use organic compost monthly.",
            "🌞 Ensure full sunlight for 6–8 hours daily."
        ]
    },

    # -------- POTATO --------
    "Potato___Early_blight": {
        "type": "Infected",
        "description": "Caused by *Alternaria solani*. Creates brown concentric spots on older leaves, often with yellow halos.",
        "prescription": [
            "🌿 Remove affected leaves regularly.",
            "🧴 Use fungicides containing mancozeb or copper.",
            "🚜 Rotate crops to prevent reinfection."
        ]
    },
    "Potato___Late_blight": {
        "type": "Infected",
        "description": "Caused by *Phytophthora infestans*. Produces dark lesions that can destroy foliage rapidly.",
        "prescription": [
            "⚠️ Destroy severely infected plants.",
            "💧 Avoid overhead watering.",
            "🧴 Spray fungicides containing metalaxyl or chlorothalonil."
        ]
    },
    "Potato___healthy": {
        "type": "Healthy",
        "description": "Your potato plant looks strong and disease-free.",
        "prescription": [
            "✅ Maintain proper irrigation and soil drainage.",
            "🌞 Ensure 6–8 hours of sunlight.",
            "🌿 Apply balanced fertilizer during growth."
        ]
    },

    # -------- TOMATO --------
    "Tomato_Bacterial_spot": {
        "type": "Infected",
        "description": "Caused by *Xanthomonas vesicatoria*. Results in small brown spots on leaves, stems, and fruits.",
        "prescription": [
            "⚠️ Remove infected leaves immediately.",
            "🧴 Apply copper-based sprays weekly.",
            "🚜 Rotate crops every season."
        ]
    },
    "Tomato_Early_blight": {
        "type": "Infected",
        "description": "Caused by *Alternaria solani*. Appears as dark spots with concentric rings on older leaves.",
        "prescription": [
            "🌿 Remove lower infected leaves.",
            "🧴 Spray neem oil or chlorothalonil fungicide.",
            "🚜 Avoid planting tomatoes in the same soil each year."
        ]
    },
    "Tomato_Late_blight": {
        "type": "Infected",
        "description": "Caused by *Phytophthora infestans*. Shows irregular, water-soaked spots on leaves and fruit decay.",
        "prescription": [
            "⚠️ Remove infected parts immediately.",
            "💧 Avoid wetting leaves during irrigation.",
            "🧴 Apply fungicides with mancozeb or copper hydroxide."
        ]
    },
    "Tomato_Leaf_Mold": {
        "type": "Infected",
        "description": "Caused by *Passalora fulva*. Yellow spots on upper surfaces with grayish mold beneath.",
        "prescription": [
            "🧼 Remove affected leaves.",
            "🌬️ Improve greenhouse air circulation.",
            "🧴 Apply fungicide (chlorothalonil or copper-based)."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "type": "Infected",
        "description": "Caused by *Septoria lycopersici*. Small circular spots with dark borders appear on lower leaves.",
        "prescription": [
            "🌿 Prune and destroy infected leaves.",
            "🧴 Apply fungicides like mancozeb or chlorothalonil.",
            "🚜 Rotate crops regularly."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "type": "Infected",
        "description": "Caused by *Tetranychus urticae* mites. Leaves turn yellow and develop fine webbing.",
        "prescription": [
            "🧴 Spray neem oil or insecticidal soap.",
            "💦 Keep humidity high to reduce mite spread.",
            "🌿 Remove heavily infested leaves."
        ]
    },
    "Tomato__Target_Spot": {
        "type": "Infected",
        "description": "Caused by *Corynespora cassiicola*. Produces concentric brown spots on leaves and fruits.",
        "prescription": [
            "🌿 Remove infected leaves promptly.",
            "🧴 Apply fungicide sprays every 7–10 days.",
            "🌞 Ensure proper ventilation and dry leaves."
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "type": "Infected",
        "description": "A viral disease transmitted by whiteflies causing curling and yellowing of leaves.",
        "prescription": [
            "🪰 Control whiteflies using sticky traps.",
            "🧴 Apply insecticidal soap or neem oil.",
            "🚜 Remove and destroy infected plants."
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "type": "Infected",
        "description": "A viral infection causing mottled, light-green patches and distorted leaves.",
        "prescription": [
            "⚠️ Remove infected plants immediately.",
            "🧴 Disinfect tools and avoid handling healthy plants after infected ones.",
            "🌿 Grow resistant tomato varieties."
        ]
    },
    "Tomato_healthy": {
        "type": "Healthy",
        "description": "Your tomato plant appears healthy and free from disease.",
        "prescription": [
            "✅ Maintain proper watering schedule.",
            "🌞 Ensure full sunlight and air circulation.",
            "🌿 Apply organic fertilizers for steady growth."
        ]
    }
}

# ----------------------------------------------------
# 🔍 Prediction Section
# ----------------------------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Leaf Image", use_container_width=True)
    st.write("⏳ Analyzing the image... Please wait.")

    model = load_trained_model()
    class_names = load_class_names()
    img_array = preprocess_image(uploaded_file)
    preds = model.predict(img_array)[0]

    # Top-3 Predictions
    top_indices = preds.argsort()[-3:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_conf = [preds[i] * 100 for i in top_indices]

    predicted_class = top_classes[0]
    main_conf = top_conf[0]

    result = disease_info.get(predicted_class)

    # Display prediction result
    if result:
        if result["type"] == "Healthy":
            st.success(f"🌱 **Prediction:** {predicted_class.replace('_', ' ')} (Healthy)")
            st.info(f"✅ Confidence: {main_conf:.2f}%")
            st.markdown(f"**🩺 Description:** {result['description']}")
            st.markdown("**🌿 Maintenance Tips:**")
            for tip in result["prescription"]:
                st.markdown(f"- {tip}")
            st.balloons()
        else:
            st.error(f"🚨 **Prediction:** {predicted_class.replace('_', ' ')} (Infected)")
            st.warning(f"🧬 Confidence: {main_conf:.2f}%")
            st.markdown(f"**🦠 Disease Info:** {result['description']}")
            st.markdown("**💊 Suggested Actions:**")
            for tip in result["prescription"]:
                st.markdown(f"- {tip}")
    else:
        st.warning(f"⚠️ {predicted_class} detected, but no detailed info found.")
        st.markdown("🧾 **General Advice:**")
        st.markdown("- Isolate the infected plant.")
        st.markdown("- Ensure good airflow and proper sunlight.")
        st.markdown("- Apply organic fungicide weekly.")

    # 📊 Visualization of top-3 predictions
    st.markdown("### 📊 Top 3 Predictions")
    chart_data = pd.DataFrame({'Disease': top_classes, 'Confidence (%)': top_conf})
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(chart_data['Disease'], chart_data['Confidence (%)'], color='seagreen')
    ax.invert_yaxis()
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Prediction Confidence Levels')
    st.pyplot(fig)

else:
    st.info("👆 Upload a clear plant leaf image to start the analysis.")

# ----------------------------------------------------
# 🧭 Footer
# ----------------------------------------------------
st.markdown("---")
st.caption("🌿 *AI-powered Plant Health Detection | Built with ❤️ by Mukul Rajput*")

# ----------------------------------------------------
# 🌟 Extra Beautification & Visualization Enhancements
# ----------------------------------------------------
st.markdown(
    """
    <style>
    /* 🌑 Dark theme base */
    body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }

    /* 🌿 Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #58a6ff !important;
        font-weight: 700;
    }

    p, div, span {
        color: #c9d1d9 !important;
    }

    /* 🌿 Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ea043, #3fb950);
        transform: scale(1.05);
        color: #ffffff;
    }

    /* 🌱 Metric Cards */
    [data-testid="stMetricValue"] {
        color: #79c0ff !important;
        font-size: 1.6rem;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
    }

    /* 🌿 Boxes and sections */
    .result-box, .chart-box, .suggestion-box {
        background-color: #161b22;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
        padding: 20px;
        border: 1px solid #30363d;
        margin-top: 20px;
    }

    /* 🌿 Progress bar */
    .stProgress > div > div > div > div {
        background-color: #3fb950;
    }

    /* 🌿 Expander styling */
    [data-testid="stExpander"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px;
    }
    [data-testid="stExpander"] div p {
        color: #c9d1d9 !important;
    }

    /* 🌿 Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
        border-right: 1px solid #30363d;
    }

    /* 🌿 Divider and line */
    hr, .stMarkdown hr {
        border: 1px solid #30363d !important;
    }

    /* 🌿 Links */
    a {
        color: #58a6ff !important;
        text-decoration: none;
    }
    a:hover {
        color: #79c0ff !important;
        text-decoration: underline;
    }

    /* 🌿 File uploader */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 2px dashed #30363d;
        border-radius: 10px;
        padding: 10px;
    }

    /* 🌿 Dropdowns */
    [data-baseweb="select"] {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
    }

    /* 🌿 Tooltips and hover effects */
    .tooltip {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    /* 🌿 Footer */
    footer {
        background-color: #0d1117 !important;
        color: #8b949e !important;
        border-top: 1px solid #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# 🌙 Enhanced Metric Section for Dark Mode
# ----------------------------------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 Classifying...")

    model = load_trained_model()
    class_names = load_class_names()
    img_array = preprocess_image(uploaded_file)
    preds = model.predict(img_array)

    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    predicted_label = class_names[predicted_class]
    health_status = "Healthy" if "healthy" in predicted_label.lower() else "Infected"

    # ---- Custom Dark Theme Style ----
    st.markdown("""
    <style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #3f3f3f;
        border-radius: 15px;
        padding: 18px;
        margin: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        color: #f5f5f5;
    }
    .metric-card:hover {
        box-shadow: 0 6px 18px rgba(50,205,50,0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 16px;
        color: #a3a3a3;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 18px;
        font-weight: 700;
        color: #80ff80;
    }
    .metric-subtext {
        font-size: 14px;
        color: #b3b3b3;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Display Results ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🧠 Prediction Confidence</div>
            <div class="metric-value">{confidence * 100:.2f}%</div>
            <div class="metric-subtext">Model Accuracy Level</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌿 Health Status</div>
            <div class="metric-value">{health_status}</div>
            <div class="metric-subtext">AI Diagnosis</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📋 Top Class</div>
            <div class="metric-value">{predicted_label}</div>
            <div class="metric-subtext">Detected Disease</div>
        </div>
        """, unsafe_allow_html=True)

    # 🎯 Show result summary
    st.success(f"**Prediction:** {predicted_label}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")

    # 🎈 Balloons on success
    st.balloons()


# ----------------------------------------------------
# 🌼 Add Disease Info Expansion Section
# ----------------------------------------------------
with st.expander("📚 Learn More About Detected Diseases"):
    if 'predicted_class' in locals():
        st.markdown(f"### 🧬 {class_names[int(predicted_class)].replace('_', ' ')}")
        if result:
            st.write(f"**Type:** {result['type']}")
            st.write(f"**Description:** {result['description']}")
            st.write("**🩺 Health Tips / Remedies:**")
            for tip in result['prescription']:
                st.markdown(f"- {tip}")
        else:
            st.info("ℹ️ No detailed data available for this class yet.")
    else:
        st.info("Upload an image first to view detailed info.")

# ----------------------------------------------------
# 📈 Add Animated Confidence Visualization
# ----------------------------------------------------
if 'top_conf' in locals():
    import plotly.express as px
    df_conf = pd.DataFrame({
        'Disease': top_classes,
        'Confidence': top_conf
    })
    fig = px.bar(
        df_conf,
        x='Disease',
        y='Confidence',
        color='Confidence',
        color_continuous_scale='greens',
        title='🌱 Interactive Prediction Confidence Chart',
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# 🌙 Add Final Message Section (Dark Theme)
# ----------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="
    text-align:center;
    font-size:18px;
    padding:15px;
    background-color:#161b22;
    color:#e6edf3;
    border-radius:12px;
    border:1px solid #30363d;
    box-shadow: 0 0 15px rgba(63,185,80,0.2);
">
🌿 <b>Tip:</b> Keep an eye on your plants — early detection ensures healthy growth!<br>
🍀 Built with ❤️ by <b style='color:#3fb950;'>Mukul Rajput</b> using 
<b style='color:#58a6ff;'>TensorFlow</b> & <b style='color:#f0883e;'>Streamlit</b>.
</div>
""", unsafe_allow_html=True)
