# import streamlit as st
# from PIL import Image
# import numpy as np
# import cv2
# import tensorflow as tf
# import random

# # Page config
# st.set_page_config(page_title="Face Recognition — But Make It Fun 😎", layout="centered")

# # Title
# st.title("🧠 Face Recognition With Attitude")
# st.caption("Not just a model... I'm judging your face with style.")

# # Load model
# model = tf.keras.models.load_model("my_model.h5")

# # Face personas
# personas = [
#     "The Overthinker",
#     "The Last-Minute Email Checker",
#     "The 'Just Woke Up' Face",
#     "The Snack Ninja",
#     "The 'I Swear I'm Not Tired' Look",
#     "The Zoom Meeting Legend",
#     "The Mysterious Stranger",
#     "The Chill One",
#     "The Person Who Just Blinks in Group Photos",
#     "The Face That Knows Too Much"
# ]

# # Upload image
# uploaded_file = st.file_uploader("Upload your photo and let's see what you're hiding 🧐", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="📷 Nice mugshot!", use_column_width=True)

#     # Process
#     img_np = np.array(image)
#     img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#     img_resized = cv2.resize(img, (100, 100)) / 255.0
#     img_input = np.expand_dims(img_resized, axis=0)

#     prediction = model.predict(img_input)
#     prob = prediction[0][0]

#     st.markdown("---")
    
#     # Choose a random persona regardless of output
#     persona = random.choice(personas)

#     if prob < 0.5:
#         st.success("✅ You’ve been recognized. Welcome back, legend.")
#         st.markdown(f"🕵️ Identity match: **{persona}**")
#         st.progress(100, text="Confidence: 100% (Or close enough)")
#     else:
#         st.warning("❓ Unrecognized. This face confuses even my deep layers.")
#         st.markdown(f"🧩 You might be... **{persona}** — or someone in disguise.")
#         st.slider("Confidence (self-rated)", 0, 100, value=72, help="Because self-esteem matters too 😌")

#     # Mystery message
#     mystery = random.choice([
#         "🧃 Fun Fact: 92% of face models have never seen a real pineapple.",
#         "🔒 Your photo has not been sold to any aliens. Yet.",
#         "🐍 This model was not trained using snakes. Probably.",
#         "📼 This session will self-destruct in 30 minutes. Just kidding.",
#         "🧠 I’m basically a face psychic with less fraud."
#     ])
#     st.markdown("---")
#     st.info(mystery)

#     # Retry
#     if st.button("Try another face"):
#         st.experimental_rerun()

# else:
#     st.markdown("👆 Go ahead, upload a face. Don't worry — it won't bite.")



import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import random
import os
import requests

# ---- Download model from Google Drive if not present ----
model_url = "https://drive.google.com/uc?export=download&id=1XRIEIVjFLahkzVMNgTLLp9KTuQVmRpTn"
model_path = "my_model.h5"

if not os.path.exists(model_path):
    with st.spinner("📦 Downloading face recognition model..."):
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)

# ---- Load the model ----
model = tf.keras.models.load_model(model_path)

# ---- UI Elements ----
st.title("Face-ception: AI Knows Beauty When It Sees It")
st.caption("Not just a model... I'm judging your face with style.")

personas = [
    "The Overthinker", "The Last-Minute Email Checker", "The 'Just Woke Up' Face",
    "The Snack Ninja", "The 'I Swear I'm Not Tired' Look", "The Zoom Meeting Legend",
    "The Mysterious Stranger", "The Chill One", "The Person Who Just Blinks in Group Photos",
    "The Face That Knows Too Much"
]

compliments = [
    "✨ Absolute icon. The algorithm just blushed.",
    "Face detected: statistically flawless.",
    "Mirror: shattered from envy.",
    "If facial recognition had favorites, it would be you.",
    "AI says: Wow. Just wow.",
    "🎯 Target locked: Absolute visual excellence.",
    "💫 Some faces are made for deep learning. Yours rewrote the layers.",
    "🌸 Eyelids like whispers, symmetry like sonnets. You didn’t wake up like this—you evolved.",
    "🎨 If beauty were a dataset, you’d be the anomaly poets whisper about.",
    "🧠 AI says: 'Finally, someone worthy of my neurons.'"
]

roasts = [
    "Detected: suspiciously normal.",
    "Hmm. This face triggers our sarcasm module.",
    "Are you sure this isn’t AI-generated?",
    "Confidence: 3%. You may be a ghost.",
    "Might be a potato. But a nice potato.",
    "🥔 This face confuses pixels. Are you… a boiled vegetable?",
    "🚨 Model alert: Too symmetrical to be random, too cursed to be deliberate.",
    "📉 Beauty error 404. Please reboot your camera. Or your genetics.",
    "📸 Your photo lowered the model's accuracy. It's sending a resignation email."
]

mystery_messages = [
    "🧃 Fun Fact: 92% of face models have never seen a real pineapple.",
    "🔒 Your photo has not been sold to any aliens. Yet.",
    "🐍 This model was not trained using snakes. Probably.",
    "📼 This session will self-destruct in 30 minutes. Just kidding.",
    "🧠 I’m basically a face psychic with less fraud."
]

# ---- Upload and Predict ----
uploaded_file = st.file_uploader("Upload your photo and let's see what you're hiding 🧐", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Nice mugshot!", use_column_width=True)

    img_np = np.array(image)
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (100, 100)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_input)
    prob = prediction[0][0]

    st.markdown("---")
    persona = random.choice(personas)

    if prob < 0.5:
        st.balloons()
        st.success("🎯 Target locked: Absolute visual excellence.")
        st.markdown(f"💎 {random.choice(compliments)}")
        st.progress(100, text="Aesthetic alignment: Celestial")

    else:
        st.warning("😬 Unrecognized. Even my GPU flinched.")
        st.markdown(f"💀 {random.choice(roasts)}") 
        st.error("💀 You’ve got a face only back-end code could love.")
        st.markdown(f"🧩 You might be... **{persona}** — or someone in disguise.")
        st.caption("📝 Model Notes: *'Consider hiding behind a firewall next time.'*")
        st.slider("On a scale of 0 to fabulous, where do *you* place yourself?", 0, 100, value=random.randint(25, 60), help="Confidence is free, looks are optional.")

    st.markdown("---")
    st.info(random.choice(mystery_messages))

    

else:
    st.markdown("👆 Go ahead, upload a face. Don’t worry — it won’t bite.")
