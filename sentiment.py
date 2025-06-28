import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from underthesea import word_tokenize
from wordcloud import WordCloud
from datetime import datetime
from PIL import Image
import os
from  preprocessed import process_text, count_pos_neg_words,classify_sentiment
import nltk
import regex
from underthesea import sentiment, word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
import emoji
nltk.download('stopwords')

# Cấu hình trang
st.set_page_config(page_title="SENTIMENT ANALYSIS AND INFORMATION CLUSTERING FOR ITVIEC", layout="wide")

# Tải dữ liệu
data = pd.read_csv('final_data.csv')


# Thanh bên (Sidebar)
st.sidebar.image("channels4_banner.jpg", use_container_width=True)
st.title("ITVIEC SENTIMENT ANALYSIS AND CLUSTERING")

# Menu
menu = ["Business Objective","Build Project","Sentiment Analysis & Clustering"]
choice = st.sidebar.selectbox('Menu', menu)

st.sidebar.write("""#### 🟢 Thành viên thực hiện: 
    Trương Văn Lê
    email: truongvanle999@gmail.com
    Đào Tuấn Thịnh
    email: daotuanthinh@gmail.com
""")

st.sidebar.write("""#### 🟡 Giáo viên hướng dẫn: 
    Cô Khuất Thuỳ Phương""")

current_time = datetime.now()
st.sidebar.write(f"""#### 🗓️ Thời gian báo cáo:
    {current_time.strftime('%d-%m-%Y %H:%M %Z')}""") 




# Nội dung theo lựa chọn menu
if choice == 'Business Objective':    
    st.subheader("Business Objective") 
    st.markdown("""
    Xây dựng hệ thống kiểm tra cảm xúc của người dùng về công ty, phân loại đánh giá là tích cực, tiêu cực hay trung tính và so sánh hiệu quả của các mô hình học máy.
    👈 **Chọn phần khác trên thanh bên** để xem chi tiết.
    ### Hệ thống bao gồm:
    - Phân tích cảm xúc với nhiều mô hình (Logistic Regression, Random Forest, LightGBM, XGBoost, Sentiment Model)
    - Phân cụm thông tin
    """)
    try:
        image1 = Image.open("Sentiment_p1.png")
        image2 = Image.open("Clustering_p1.png")
        new_size = (400, 400)
        image1_resized = image1.resize(new_size)
        image2_resized = image2.resize(new_size)
        new_image = Image.new("RGB", (800, 400), (255, 255, 255))
        new_image.paste(image1_resized, (0, 0))
        new_image.paste(image2_resized, (400, 0))
        st.image(new_image, use_container_width=True, caption="Sentiment Analysis & Clustering")
    except:
        st.error("❌ Lỗi tải hình ảnh.")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.markdown("##### 1. Chuẩn bị dữ liệu")
    st.markdown("""
    - **Đọc dữ liệu**: Tải dữ liệu từ file Reviews.xlsx và Overview_Companies.xlsx.
    - **Xử lý dữ liệu thiếu**: Loại bỏ các dòng có giá trị `null` trong các cột như `What I liked`, `Suggestions for improvement`.
    - **Kết hợp cột**: Gộp `What I liked` và `Suggestions for improvement` để xử lý trước khi huấn luyện.
    """)
    try:
        file_review = 'Reviews.xlsx'
        file_overview_company = 'Overview_Companies.xlsx'
        review = pd.read_excel(file_review)
        overview_company = pd.read_excel(file_overview_company)
        review.rename(columns={'Recommend?': 'Recommend'}, inplace=True)
        overview_company = overview_company[['Company Name', 'Company Type', 'Company size', 'Country', 'Working days', 'Overtime Policy']]
        data = pd.merge(review, overview_company, on='Company Name')
        st.dataframe(data.head(3))
        st.dataframe(data.tail(3))
    except:
        st.error("❌ Lỗi tải dữ liệu.")
    # Khám phá và tiền xủ lý dữ liệu
    st.markdown("##### 2. Khám phá và Tiền xử lý dữ liệu")
    st.markdown("""
    - Hiển thị thông tin cơ bản: Số lượng bản ghi, số lượng đặc trưng.
    - Tiền xử lý văn bản:
        1. Loại bỏ số.
        2. Loại bỏ ký tự đặc biệt.
        3. Loại bỏ stop words.
    - Hiển thị WordCloud trước và sau xử lý.
    """)
    try:
        st.write("##### WordCloud trước xử lý")
        st.image("pre_processed_text.png", use_container_width=True)
        st.write("##### WordCloud sau xử lý")
        st.image("processed_text.png", use_container_width=True)
    except:
        st.error("❌ Lỗi tải hình ảnh WordCloud.")
    # 3. Đánh giá các thuật toán
    st.markdown("##### 3. Đánh giá các thuật toán")
    st.markdown("""
    - **Danh sách các thuật toán đã triển khai**:
      - KNNBaseline
      - SVD (Singular Value Decomposition)
      - SVDpp (SVD++)
      - BaselineOnly
    - **Quy trình đánh giá**:
      - Sử dụng `cross_validate` với các chỉ số RMSE và MAE để đánh giá các thuật toán trên 5 lần gập (5-fold cross-validation).
      - Ghi lại thời gian huấn luyện và kết quả đánh giá (RMSE, MAE).
      - Kết quả được lưu vào DataFrame `results_df` để so sánh các thuật toán.
    """)
    st.image("Bảngđánhgiáthuậttoán.png", caption="Bảng đánh giá thuật toán", use_container_width=True)









elif choice == 'Sentiment Analysis & Clustering':
    st.subheader("Sentitment Analysis & Clustering")
    
    # Load model
    models_Logistic = joblib.load("modles/logistic_model.pkl")
    models_RandomForest = joblib.load("modles/random_forest_model.pkl")
    models_LightGBM = joblib.load("modles/LightGBM_model.pkl")
    models_XGBoost = joblib.load("modles/xgboost_model.pkl")

    # Models
    models = {
        "Logistic Regression": models_Logistic,
        "Random Forest": models_RandomForest,
        "LightGBM": models_LightGBM,
        "XGBoost": models_XGBoost
    }


    # Lựa chọn mô hình
    selected_model = st.radio(
        "Chọn mô hình để phân tích:",
        options=[m for m in models.keys()],
        key="model_selection",
        horizontal=True
    )

    # Chọn Công ty
    st.write('### Chọn Công ty')
    company_name = [""] + data["Company Name"].dropna().unique().tolist()
    select_company_name = st.selectbox("Chọn Công ty", company_name)
    select_company_id = None
    if select_company_name:
        select_company_id = data[data["Company Name"] == select_company_name]['id'].values[0]

    # Nhập đánh giá
    st.write("### Bước 1: Chọn điểm số (1-5)")
    salary_benefits = st.radio("💰 Salary_benefits", [1, 2, 3, 4, 5], horizontal=True, key="salary_benefits")
    training_learning = st.radio("🎓 Training_learning", [1, 2, 3, 4, 5], horizontal=True, key="training_learning")
    management_cares = st.radio("🤝 Management_cares", [1, 2, 3, 4, 5], horizontal=True, key="management_cares")
    culture_fun = st.radio("🎉 Culture_fun", [1, 2, 3, 4, 5], horizontal=True, key="culture_fun")
    office_workspace = st.radio("🏢 Office_workspace", [1, 2, 3, 4, 5], horizontal=True, key="office_workspace")

    recommend = st.selectbox("Bạn có đề xuất công ty này không?", ["Yes", "No"], index=0, key="recommend")
    # Nhập đánh giá
    st.write("### Bước 2: Viết đánh giá của bạn")
    review_text = st.text_area("📝 Ý kiến của bạn", "", key="review_text")
    
    # Chuẩn hóa phần review_text
    process_text = process_text(review_text)
    # Đếm số câu/cụm từ tích cực và tiêu cực
    pos_count, neg_count = count_pos_neg_words(review_text)

    # Đưa feature vào mô hình
    input_data = pd.DataFrame([{
         'word_count_positive': pos_count,
        'word_count_negative': neg_count,
        'Salary & benefits': salary_benefits,
        'Training & learning': training_learning,
        'Management cares about me': management_cares,
        'Culture & fun': culture_fun,
        'Office & workspace': office_workspace
}])
    # Các nút chức năng
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Dự đoán cảm xúc"):
            if selected_model == 'Random Forest':
                prediction = models_RandomForest.predict(input_data)[0]
            elif selected_model == 'Logistic Regression':
                prediction = models_Logistic.predict(input_data)[0]
            elif selected_model == 'LightGBM':
                prediction = models_LightGBM.predict(input_data)[0]
            else:
                prediction = models_XGBoost.predict(input_data)[0]
            labels = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Possitive"}
            st.success(f"✅ Sentiment predict ({selected_model}): **{labels[prediction]}**")
    # with col2:
    #     if st.button("📊 Phân cụm đánh giá"):
    #         features = np.array([[salary_benefits, training_learning, management_cares, culture_fun, office_workspace]])
    #         try:
    #             if kmeans_model is not None:
    #                 cluster_label = kmeans_model.predict(features)[0]
    #                 st.info(f"Công ty thuộc **Cụm {cluster_label}** (dựa trên điểm số)")
    #             else:
    #                 st.error("❌ Mô hình phân cụm không được tải thành công.")
    #         except Exception as e:
    #             st.error(f"❌ Lỗi khi phân cụm: {str(e)}")
    
    # with col3:
    #     if st.button("💾 Lưu đánh giá"):
    #         st.success("Đánh giá đã được lưu thành công! (giả lập)")
    
    # with col4:
    #     if st.button("🏢 Đánh giá công ty"):
    #         try:
    #             file_review = 'Reviews.xlsx'
    #             file_overview_company = 'Overview_Companies.xlsx'
    #             review = pd.read_excel(file_review)
    #             overview_company = pd.read_excel(file_overview_company)
    #             review.rename(columns={'Recommend?': 'Recommend'}, inplace=True)
    #             overview_company = overview_company[['Company Name', 'Company Type', 'Company size', 'Country', 'Working days', 'Overtime Policy']]
    #             data = pd.merge(review, overview_company, on='Company Name')
    #             st.write("**Dữ liệu đánh giá công ty**:")
    #             st.dataframe(data.head())
    #         except Exception as e:
    #             st.error(f"❌ Lỗi tải dữ liệu công ty: {str(e)}")