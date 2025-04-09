from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.header('การจำแนกข้อมูลด้วยเทคนิค ML(KNNHeart)')
col1, col2 = st.columns(2)

with col1:
   st.header("ไม่เป็นโรคหัวใจ")
   st.image("./img/img1.jpg")

with col2:
   st.header("เป็นโรคหัวใจ")
   st.image("./img/img2.jpg")



html_7 = """
<div style="background-color:#33a5ff;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ข้อมูลสำหรับทำนายโรคหัวใจ</h4></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกประเภทเป็นหรือไม่เป็นโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

pt_len = st.slider("กรุณาเลือกข้อมูล petal.length")
pt_wd = st.slider("กรุณาเลือกข้อมูล petal.width")

sp_len = st.number_input("กรุณาเลือกข้อมูล sepal.length")
sp_wd = st.number_input("กรุณาเลือกข้อมูล sepal.width")

if st.button("ทำนายผล"):
    #st.write("ทำนาย")
   dt = pd.read_csv("./data/Heart3.csv") 
   X = dt.drop('HeartDisease', axis=1)
   y = dt.HeartDisease   

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)  
    
   x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
   st.write(Knn_model.predict(x_input))
   
   out=Knn_model.predict(x_input)

   if out[0] == 'ไม่เป็นโรคหัวใจ':
    st.image("./img/img1.jpg")
   elif out[0] == 'เป็นโรคหัวใจ':       
    st.image("./img/img2.jpg")
else:
    st.write("ไม่ทำนาย")