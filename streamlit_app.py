import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Dự đoán kết quả học tập", layout="centered")
st.title(":blue[DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN]")

# =============================
# 1. Sidebar chọn loại sinh viên
# =============================
st.sidebar.subheader("Cài đặt đầu vào")
student_type = st.sidebar.selectbox("Loại sinh viên:", ("8 kỳ", "10 kỳ"))
max_semester = 6 if student_type == "8 kỳ" else 8

# Chọn kỳ hiện tại
current_semester = st.sidebar.selectbox("Kỳ hiện tại:", list(range(1, max_semester + 1)))

# Nhập GPA từ kỳ 1 đến kỳ hiện tại
gpa_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(f"GPA kỳ {i}", min_value=0.0, max_value=4.0, step=0.01, format="%.2f")
    gpa_inputs.append(gpa)

# =============================
# 2. Xử lý đầu vào và dự đoán
# =============================
if any(g == 0.0 for g in gpa_inputs):
    st.warning("⚠️ Vui lòng nhập đầy đủ GPA cho tất cả các kỳ đã chọn.")
else:
    try:
        input_data = np.array(gpa_inputs).reshape(1, -1)
        model_prefix = student_type.split()[0]  # '8' hoặc '10'

        # Định danh nhóm
        group_key = f"GPA_1_{current_semester}" if current_semester > 1 else "GPA_1"

        # =============================
        # 3. Dự đoán Final CPA
        # =============================
        cpa_model_path = f"models_streamlit/final_cpa_{model_prefix}_ki.joblib"
        cpa_dict = joblib.load(cpa_model_path)

        scaler_cpa = cpa_dict[group_key]['scaler']
        model_cpa = cpa_dict[group_key]['svr']  # hoặc 'rf'

        input_scaled_cpa = scaler_cpa.transform(input_data)
        predicted_cpa = model_cpa.predict(input_scaled_cpa)[0]

        st.subheader("🎓 Dự đoán CPA tốt nghiệp:")
        st.success(f"Final CPA: {predicted_cpa:.2f}")

        # =============================
        # 4. Dự đoán GPA kỳ tiếp theo
        # =============================
        if current_semester < max_semester:
            next_gpa_path = f"models_streamlit/next_gpa_{model_prefix}_ki.joblib"
            next_dict = joblib.load(next_gpa_path)

            scaler_next = next_dict[group_key]['scaler']
            model_next = next_dict[group_key]['svr']

            input_scaled_next = scaler_next.transform(input_data)
            predicted_next_gpa = model_next.predict(input_scaled_next)[0]

            st.subheader(f"📘 Dự đoán GPA kỳ {current_semester + 1}:")
            st.info(f"GPA dự đoán: {predicted_next_gpa:.2f}")

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi khi dự đoán: {e}")
