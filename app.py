import streamlit as st
import pandas as pd
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Mushroom Predictor", page_icon="🍄", layout="wide")

# ==========================================
# พจนานุกรมแปลรหัสตัวอักษรเป็นคำที่มนุษย์เข้าใจ
# ==========================================
mushroom_dict = {
    'cap-shape': {'b': 'Bell (ทรงระฆัง)', 'c': 'Conical (ทรงกรวย)', 'x': 'Convex (ทรงนูน)', 'f': 'Flat (ทรงแบน)', 'k': 'Knobbed (มีปุ่ม)', 's': 'Sunken (บุ๋ม)'},
    'cap-surface': {'f': 'Fibrous (มีเส้นใย)', 'g': 'Grooves (มีร่อง)', 'y': 'Scaly (เป็นเกล็ด)', 's': 'Smooth (เรียบ)'},
    'cap-color': {'n': 'Brown (น้ำตาล)', 'b': 'Buff (เหลืองหม่น)', 'c': 'Cinnamon (สีอบเชย)', 'g': 'Gray (เทา)', 'r': 'Green (เขียว)', 'p': 'Pink (ชมพู)', 'u': 'Purple (ม่วง)', 'e': 'Red (แดง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'bruises': {'t': 'Yes (มีรอยช้ำ)', 'f': 'No (ไม่มีรอยช้ำ)'},
    'odor': {'a': 'Almond (อัลมอนด์)', 'l': 'Anise (โป๊ยกั๊ก)', 'c': 'Creosote (น้ำมันทาร์)', 'y': 'Fishy (คาวปลา)', 'f': 'Foul (เหม็นเน่า)', 'm': 'Musty (อับชื้น)', 'n': 'None (ไม่มีกลิ่น)', 'p': 'Pungent (ฉุน)', 's': 'Spicy (เครื่องเทศ)'},
    'gill-size': {'b': 'Broad (กว้าง)', 'n': 'Narrow (แคบ)'},
    'stalk-shape': {'e': 'Enlarging (ขยายออก)', 't': 'Tapering (เรียวลง)'},
    # ถ้าอยากให้คอลัมน์ไหนแปลภาษาได้อีก สามารถมาเพิ่มในนี้ได้เลยครับ
}

@st.cache_resource
def load_artifacts():
    model = joblib.load('svm_mushroom_model.pkl')
    encoders = joblib.load('mushroom_encoders.pkl')
    return model, encoders

try:
    model, encoders = load_artifacts()
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบให้แน่ใจว่าวางไฟล์ .pkl ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

st.title("🍄 ระบบทำนายเห็ดพิษด้วย AI (SVM)")
st.write("เลือกคุณลักษณะของเห็ดที่คุณพบ เพื่อให้ AI ช่วยวิเคราะห์ว่าเป็นเห็ดที่ **กินได้** หรือ **มีพิษ**")
st.markdown("---")

features = [col for col in encoders.keys() if col != 'class']
user_input = {}
cols = st.columns(4)

for i, feature in enumerate(features):
    options = encoders[feature].classes_
    
    # ฟังก์ชันช่วยแปลภาษาตอนแสดงผลบน Dropdown
    def format_option(val, feat=feature):
        # ถ้ามีคำแปลในดิกชันนารี ให้แสดงคำแปล ถ้าไม่มีให้โชว์รหัสเดิมไปก่อน
        if feat in mushroom_dict and val in mushroom_dict[feat]:
            return mushroom_dict[feat][val]
        return val # fallback โชว์ตัวอักษรเดิม

    with cols[i % 4]:
        # ใช้ format_func เพื่อแปลงหน้าตาของตัวเลือกให้ดูอ่านง่าย
        user_input[feature] = st.selectbox(
            f"{feature.replace('-', ' ').title()}", 
            options, 
            format_func=format_option
        )

st.markdown("---")
if st.button("🔍 วิเคราะห์ผลลัพธ์", type="primary"):
    
    input_encoded = {}
    for feature in features:
        le = encoders[feature]
        input_encoded[feature] = le.transform([user_input[feature]])[0]
        
    input_df = pd.DataFrame([input_encoded])
    prediction = model.predict(input_df)
    
    st.markdown("### 📊 ผลการวิเคราะห์:")
    if prediction[0] == 0:
        st.success("✅ **โมเดลทำนายว่า: เห็ดชนิดนี้ กินได้ (Edible)**")
    else:
        st.error("☠️ **โมเดลทำนายว่า: เห็ดชนิดนี้ มีพิษ (Poisonous)**")
        st.warning("คำเตือน: นี่เป็นเพียงการวิเคราะห์เบื้องต้นด้วย AI ห้ามรับประทานเห็ดป่าที่ไม่รู้จักโดยเด็ดขาด!")