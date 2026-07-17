import streamlit as st
import pandas as pd
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Mushroom Predictor", page_icon="🍄", layout="wide")

# 1. โหลดโมเดลและ Encoders (ใช้ st.cache_resource เพื่อไม่ให้โหลดใหม่ทุกครั้งที่คลิก)
@st.cache_resource
def load_artifacts():
    model = joblib.load('svm_mushroom_model.pkl')
    encoders = joblib.load('mushroom_encoders.pkl')
    return model, encoders

try:
    model, encoders = load_artifacts()
except FileNotFoundError:
    st.error("ไม่พบไฟล์ .pkl กรุณาตรวจสอบให้แน่ใจว่าวางไฟล์ไว้ในโฟลเดอร์เดียวกันกับ app.py")
    st.stop()

st.title("🍄 ระบบทำนายเห็ดพิษด้วย AI (SVM)")
st.write("ระบบนี้จะประเมินลักษณะของเห็ดที่คุณพบ เพื่อตรวจสอบว่ามัน **กินได้ (Edible)** หรือ **มีพิษ (Poisonous)**")

# 2. สร้างฟอร์มรับข้อมูล (ดึงฟีเจอร์และตัวเลือกมาจาก Encoders อัตโนมัติ)
st.header("โปรดเลือกลักษณะของเห็ด")
st.info("ตัวอักษรในตัวเลือกคือรหัสลักษณะของเห็ดตาม Dataset ต้นฉบับ (เช่น b=bell, c=conical)")

# ดึงรายชื่อคอลัมน์ทั้งหมด ยกเว้น 'class' ซึ่งเป็นคำตอบ
features = [col for col in encoders.keys() if col != 'class']

user_input = {}

# แบ่งหน้าจอเป็น 4 คอลัมน์เพื่อให้ฟอร์มไม่ยาวจนเกินไป
cols = st.columns(4)

# วนลูปสร้าง Dropdown (Selectbox) สำหรับทุกฟีเจอร์
for i, feature in enumerate(features):
    options = encoders[feature].classes_
    with cols[i % 4]:
        # บันทึกค่าที่ผู้ใช้เลือกลงใน Dictionary
        user_input[feature] = st.selectbox(f"{feature}", options)

# 3. ส่วนของการประมวลผลเมื่อกดปุ่ม
st.markdown("---")
if st.button("🔍 วิเคราะห์ผลลัพธ์", type="primary"):
    
    # แปลงข้อมูลตัวอักษรที่ผู้ใช้เลือก ให้กลับไปเป็นตัวเลขด้วย Encoder ตัวเดิม
    input_encoded = {}
    for feature in features:
        le = encoders[feature]
        # ต้องครอบด้วย List ก่อน transform และดึง index [0] ออกมา
        input_encoded[feature] = le.transform([user_input[feature]])[0]
        
    # จัดเรียงข้อมูลให้เป็นตาราง (DataFrame) 1 แถวเพื่อป้อนเข้าโมเดล
    input_df = pd.DataFrame([input_encoded])
    
    # ให้โมเดล SVM ทำนาย
    prediction = model.predict(input_df)
    
    # 0 = Edible, 1 = Poisonous (อ้างอิงจากตอนที่เรา Label Encode คอลัมน์ class)
    st.markdown("### ผลการวิเคราะห์:")
    if prediction[0] == 0:
        st.success("✅ **โมเดลทำนายว่า: เห็ดชนิดนี้ กินได้ (Edible)**")
    else:
        st.error("☠️ **โมเดลทำนายว่า: เห็ดชนิดนี้ มีพิษ (Poisonous)**")
        st.warning("คำเตือน: นี่เป็นเพียงการวิเคราะห์เบื้องต้นด้วย AI ห้ามรับประทานเห็ดป่าที่ไม่รู้จักโดยเด็ดขาด!")