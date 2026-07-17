import streamlit as st
import pandas as pd
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Mushroom Predictor", page_icon="🍄", layout="wide")

# ==========================================
# พจนานุกรมแปลรหัสตัวอักษรเป็นคำที่มนุษย์เข้าใจ (ครบทั้ง 22 ฟีเจอร์)
# ==========================================
mushroom_dict = {
    'cap-shape': {'b': 'Bell (ทรงระฆัง)', 'c': 'Conical (ทรงกรวย)', 'x': 'Convex (ทรงนูน)', 'f': 'Flat (ทรงแบน)', 'k': 'Knobbed (มีปุ่ม)', 's': 'Sunken (บุ๋ม)'},
    'cap-surface': {'f': 'Fibrous (มีเส้นใย)', 'g': 'Grooves (มีร่อง)', 'y': 'Scaly (เป็นเกล็ด)', 's': 'Smooth (เรียบ)'},
    'cap-color': {'n': 'Brown (น้ำตาล)', 'b': 'Buff (เหลืองหม่น)', 'c': 'Cinnamon (สีอบเชย)', 'g': 'Gray (เทา)', 'r': 'Green (เขียว)', 'p': 'Pink (ชมพู)', 'u': 'Purple (ม่วง)', 'e': 'Red (แดง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'bruises': {'t': 'Yes (มีรอยช้ำ)', 'f': 'No (ไม่มีรอยช้ำ)'},
    'odor': {'a': 'Almond (อัลมอนด์)', 'l': 'Anise (โป๊ยกั๊ก)', 'c': 'Creosote (น้ำมันทาร์)', 'y': 'Fishy (คาวปลา)', 'f': 'Foul (เหม็นเน่า)', 'm': 'Musty (อับชื้น)', 'n': 'None (ไม่มีกลิ่น)', 'p': 'Pungent (ฉุน)', 's': 'Spicy (เครื่องเทศ)'},
    'gill-attachment': {'a': 'Attached (ติดก้าน)', 'd': 'Descending (ไหลลง)', 'f': 'Free (อิสระ)', 'n': 'Notched (เป็นรอยบาก)'},
    'gill-spacing': {'c': 'Close (ชิด)', 'w': 'Crowded (เบียด)', 'd': 'Distant (ห่าง)'},
    'gill-size': {'b': 'Broad (กว้าง)', 'n': 'Narrow (แคบ)'},
    'gill-color': {'k': 'Black (ดำ)', 'n': 'Brown (น้ำตาล)', 'b': 'Buff (เหลืองหม่น)', 'h': 'Chocolate (ช็อกโกแลต)', 'g': 'Gray (เทา)', 'r': 'Green (เขียว)', 'o': 'Orange (ส้ม)', 'p': 'Pink (ชมพู)', 'u': 'Purple (ม่วง)', 'e': 'Red (แดง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'stalk-shape': {'e': 'Enlarging (ขยายออก)', 't': 'Tapering (เรียวลง)'},
    'stalk-root': {'b': 'Bulbous (กระเปาะ)', 'c': 'Club (กระบอง)', 'u': 'Cup (ถ้วย)', 'e': 'Equal (เท่ากัน)', 'z': 'Rhizomorphs (คล้ายราก)', 'r': 'Rooted (หยั่งราก)', '?': 'Missing (ไม่ทราบข้อมูล)'},
    'stalk-surface-above-ring': {'f': 'Fibrous (มีเส้นใย)', 'y': 'Scaly (เป็นเกล็ด)', 'k': 'Silky (มันวาว)', 's': 'Smooth (เรียบ)'},
    'stalk-surface-below-ring': {'f': 'Fibrous (มีเส้นใย)', 'y': 'Scaly (เป็นเกล็ด)', 'k': 'Silky (มันวาว)', 's': 'Smooth (เรียบ)'},
    'stalk-color-above-ring': {'n': 'Brown (น้ำตาล)', 'b': 'Buff (เหลืองหม่น)', 'c': 'Cinnamon (สีอบเชย)', 'g': 'Gray (เทา)', 'o': 'Orange (ส้ม)', 'p': 'Pink (ชมพู)', 'e': 'Red (แดง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'stalk-color-below-ring': {'n': 'Brown (น้ำตาล)', 'b': 'Buff (เหลืองหม่น)', 'c': 'Cinnamon (สีอบเชย)', 'g': 'Gray (เทา)', 'o': 'Orange (ส้ม)', 'p': 'Pink (ชมพู)', 'e': 'Red (แดง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'veil-type': {'p': 'Partial (บางส่วน)', 'u': 'Universal (ทั้งหมด)'},
    'veil-color': {'n': 'Brown (น้ำตาล)', 'o': 'Orange (ส้ม)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'ring-number': {'n': 'None (ไม่มี)', 'o': 'One (หนึ่งวง)', 't': 'Two (สองวง)'},
    'ring-type': {'c': 'Cobwebby (ใยแมงมุม)', 'e': 'Evanescent (เลือนหาย)', 'f': 'Flaring (บานออก)', 'l': 'Large (ใหญ่)', 'n': 'None (ไม่มี)', 'p': 'Pendant (ห้อยย้อย)', 's': 'Sheathing (หุ้ม)', 'z': 'Zone (เป็นวง)'},
    'spore-print-color': {'k': 'Black (ดำ)', 'n': 'Brown (น้ำตาล)', 'b': 'Buff (เหลืองหม่น)', 'h': 'Chocolate (ช็อกโกแลต)', 'r': 'Green (เขียว)', 'o': 'Orange (ส้ม)', 'u': 'Purple (ม่วง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'population': {'a': 'Abundant (ชุกชุม)', 'c': 'Clustered (ขึ้นเป็นกลุ่มก้อน)', 'n': 'Numerous (จำนวนมาก)', 's': 'Scattered (กระจายตัว)', 'v': 'Several (หลายดอก)', 'y': 'Solitary (ขึ้นดอกเดี่ยว)'},
    'habitat': {'g': 'Grasses (หญ้า)', 'l': 'Leaves (กองใบไม้)', 'm': 'Meadows (ทุ่งหญ้า)', 'p': 'Paths (ตามทางเดิน)', 'u': 'Urban (ในเมือง)', 'w': 'Waste (กองขยะ)', 'd': 'Woods (ในป่า)'}
}

@st.cache_resource
def load_artifacts():
    model = joblib.load('svm_mushroom_model.pkl')
    encoders = joblib.load('mushroom_encoders.pkl')
    return model, encoders

try:
    model, encoders = load_artifacts()
except FileNotFoundError:
    st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบให้แน่ใจว่าวางไฟล์ .pkl ไว้ในโฟลเดอร์เดียวกันกับ app.py")
    st.stop()

st.title("🍄 ระบบทำนายเห็ดพิษด้วย AI (SVM)")
st.write("เลือกลักษณะต่างๆ ของเห็ดที่คุณพบ เพื่อให้ AI ช่วยวิเคราะห์ว่าเป็นเห็ดที่ **กินได้** หรือ **มีพิษ**")
st.markdown("---")

# ดึงชื่อคอลัมน์ที่ไม่ใช่ 'class'
features = [col for col in encoders.keys() if col != 'class']
user_input = {}
cols = st.columns(4)

for i, feature in enumerate(features):
    options = encoders[feature].classes_
    
    # ฟังก์ชันช่วยแปลภาษา
    def format_option(val, feat=feature):
        if feat in mushroom_dict and val in mushroom_dict[feat]:
            return mushroom_dict[feat][val]
        return val

    with cols[i % 4]:
        # จัดชื่อหัวข้อให้สวยงาม (ตัดขีดกลางออก และทำตัวพิมพ์ใหญ่คำแรก)
        display_name = feature.replace('-', ' ').title()
        user_input[feature] = st.selectbox(
            display_name, 
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
        st.warning("⚠️ คำเตือน: นี่เป็นเพียงการวิเคราะห์เบื้องต้นด้วย AI ห้ามรับประทานเห็ดป่าที่ไม่รู้จักโดยเด็ดขาด!")