import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# قائمة بأسماء الأعمدة
column_names = [
    "Sample_code_number", "Clump_Thickness", "Uniformity_of_Cell_Size",
    "Uniformity_of_Cell_Shape", "Marginal_Adhesion", "Single_Epithelial_Cell_Size",
    "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class",
]

# بيانات الخصائص
feature_info = {
    "Clump_Thickness": {
        "arabic_name": "سمك الكتلة",
        "description": "هذا المقياس يحدد مدى سماكة أو تراص الخلايا في العينة.",
        "impact": "كلما زادت القيمة، زادت احتمالية أن يكون الورم خبيثًا، حيث أن الخلايا السرطانية تكون أكثر تراصًا."
    },
    "Uniformity_of_Cell_Size": {
        "arabic_name": "انتظام حجم الخلية",
        "description": "تقيّم هذه الخاصية مدى التباين في أحجام خلايا العينة.",
        "impact": "الخلايا السرطانية غالبًا ما تكون غير منتظمة في الحجم. القيم الأعلى تشير إلى زيادة احتمالية أن يكون الورم خبيثًا."
    },
    "Uniformity_of_Cell_Shape": {
        "arabic_name": "انتظام شكل الخلية",
        "description": "يقيس مدى التباين في شكل الخلايا.",
        "impact": "الخلايا السرطانية تظهر أشكالًا غير منتظمة. القيم الأعلى تزيد من احتمالية أن يكون الورم خبيثًا."
    },
    "Marginal_Adhesion": {
        "arabic_name": "الالتصاق الهامشي",
        "description": "يقيس مدى التصاق الخلايا ببعضها البعض عند حافة الكتلة.",
        "impact": "الخلايا الخبيثة غالبًا ما تفقد قدرتها على الالتصاق ببعضها. القيم الأعلى تشير إلى الالتصاق الضعيف، مما يزيد من احتمالية أن يكون الورم خبيثًا."
    },
    "Single_Epithelial_Cell_Size": {
        "arabic_name": "حجم الخلية الظهارية",
        "description": "تقيس هذه الخاصية حجم الخلايا الظهارية الفردية الموجودة في العينة.",
        "impact": "الخلايا السرطانية تميل إلى أن تكون أكبر حجمًا. القيم الأعلى تزيد من احتمالية أن يكون الورم خبيثًا."
    },
    "Bare_Nuclei": {
        "arabic_name": "النواة العارية",
        "description": "تقيّم هذه الخاصية عدد النوى التي انفصلت عن غشاء الخلية المحيط بها.",
        "impact": "وجود عدد كبير من النوى العارية يعد مؤشرًا قويًا على الخلايا السرطانية. القيم الأعلى تزيد بشكل كبير من احتمالية أن يكون الورم خبيثًا."
    },
    "Bland_Chromatin": {
        "arabic_name": "الكروماتين الأملس",
        "description": "الكروماتين هو المادة الوراثية في نواة الخلية. هذه الخاصية تقيّم مدى نعومته أو خشونته.",
        "impact": "الكروماتين الخشن وغير المنتظم يعد مؤشراً على الخلايا الخبيثة. القيم الأعلى تزيد من احتمالية أن يكون الورم خبيثًا."
    },
    "Normal_Nucleoli": {
        "arabic_name": "النويّات الطبيعية",
        "description": "النويّات هي هياكل صغيرة داخل نواة الخلية. هذه الخاصية تقيّم حجمها وشكلها.",
        "impact": "الخلايا السرطانية غالبًا ما تحتوي على نويات أكبر حجمًا وغير منتظمة. القيم الأعلى (التي تشير إلى أنها أقل طبيعية) تزيد من احتمالية أن يكون الورم خبيثًا."
    },
    "Mitoses": {
        "arabic_name": "الانقسام الفتيلي",
        "description": "يقيس هذا المقياس عدد الخلايا التي تمر بعملية الانقسام الفتيلي.",
        "impact": "الخلايا السرطانية تنقسم بمعدل أعلى بكثير من الخلايا الطبيعية. القيم الأعلى تزيد من احتمالية أن يكون الورم خبيثًا."
    }
}


# تعيين إعدادات الصفحة
st.set_page_config(
    page_title="تطبيق التنبؤ بالسرطان",
    layout="wide",
    initial_sidebar_state="expanded",
)

# تحميل النموذج المحفوظ
try:
    model = joblib.load('cancer_prediction_model.pkl')
except FileNotFoundError:
    st.error("لم يتم العثور على ملف النموذج 'cancer_prediction_model.pkl'. يرجى التأكد من وجوده في نفس المجلد.")

# دالة لرسم حدود قرار النموذج
def plot_decision_boundary(model, X, y):
    feature1_name = "Uniformity_of_Cell_Size"
    feature2_name = "Clump_Thickness"
    
    X_plot = X[[feature1_name, feature2_name]]
    y_plot = y

    x_min, x_max = X_plot.iloc[:, 0].min() - 1, X_plot.iloc[:, 0].max() + 1
    y_min, y_max = X_plot.iloc[:, 1].min() - 1, X_plot.iloc[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z_input = np.c_[xx.ravel(), yy.ravel()]
    dummy_features = np.zeros((Z_input.shape[0], 7))
    Z_full_input = np.concatenate([Z_input, dummy_features], axis=1)
    
    Z = model.predict(Z_full_input)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X_plot.iloc[:, 0], X_plot.iloc[:, 1], c=y_plot, s=20, edgecolor='k', cmap='viridis')
    plt.xlabel(f"({feature1_name}) انتظام حجم الخلية")
    plt.ylabel(f"({feature2_name}) سمك الكتلة")
    plt.title("حدود قرار النموذج")
    plt.legend(['حميد', 'خبيث'], loc='upper right')
    return plt

# اختيار الصفحة من الشريط الجانبي
st.sidebar.title("قائمة التنقل")
page = st.sidebar.selectbox("اختر الصفحة", ["الرئيسية", "التنبؤ", "معلومات عن الخصائص"])

# --- واجهة الترحيب (الرئيسية) ---
if page == "الرئيسية":
    st.markdown("<h1 style='text-align: center;'>تطبيق التنبؤ بالسرطان</h1>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1549487779-111af34c9c1b?q=80&w=1470&auto=format&fit=crop")
    st.markdown("---")
    
    st.markdown("""
        <div style='
            background-color: #E6F3FF; 
            border: 1px solid #B3D8FF; 
            border-radius: 5px; 
            padding: 10px; 
            text-align: center;
        '>
            <h3 style='color: #004085;'>مرحبًا بك في هذا التطبيق المبتكر ⚕️</h3>
            <p style='color: #004085;'>هذه الأداة الذكية مبنية على التعلم الآلي لتقديم <strong>تنبؤ أولي</strong> حول طبيعة الورم (حميد أو خبيث) بناءً على خصائص معينة للخلايا.</p>
            <p style='color: #004085;'><strong>ملاحظة هامة:</strong> هذا التطبيق أداة مساعدة بحثية وليس بديلاً عن التشخيص الطبي المتخصص.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("<h3 style='text-align: center;'>عن المالك</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>تم تطوير هذا التطبيق بواسطة: <strong>نورا الحرازي</strong></p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("#### للذهاب إلى صفحة التنبؤ من الشريط الجانبي:")
    # if st.button("بدء التنبؤ ➡️", use_container_width=True):
    #     st.session_state.page = "التنبؤ"
    #     st.experimental_rerun()

# --- واجهة إدخال البيانات والتنبؤ ---
elif page == "التنبؤ":
    st.markdown("<h1 style='text-align: center;'>صفحة التنبؤ</h1>", unsafe_allow_html=True)
    
    with st.expander("❤️ رسائل ونصائح إيجابية ❤️", expanded=True):
        st.markdown("""
            * **القوة في المعرفة:** هذا التطبيق يمنحك أداة لفهم البيانات، لكن تذكر دائمًا أنه ليس بديلاً عن استشارة الطبيب المختص.
            * **الأمل دائمًا موجود:** التقدم في مجال الطب يفتح آفاقًا جديدة للعلاج.
        """)
    st.markdown("---")
    
    st.write("#### أدخل خصائص العينة:")

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("##### الخصائص الأساسية")
            clump_thickness = st.slider("سمك الكتلة", 1, 10, 5)
            cell_size = st.slider("انتظام حجم الخلية", 1, 10, 5)
            cell_shape = st.slider("انتظام شكل الخلية", 1, 10, 5)
            marginal_adhesion = st.slider("الالتصاق الهامشي", 1, 10, 5)
            single_epithelial_cell_size = st.slider("حجم الخلية الظهارية", 1, 10, 5)
        with col2:
            st.write("##### الخصائص المجهرية")
            bare_nuclei = st.slider("النواة العارية", 1, 10, 5)
            bland_chromatin = st.slider("الكروماتين الأملس", 1, 10, 5)
            normal_nucleoli = st.slider("النويّات الطبيعية", 1, 10, 5)
            mitoses = st.slider("الانقسام الفتيلي", 1, 10, 5)

    st.markdown("---")
    if st.button("التنبؤ", use_container_width=True, type="primary"):
        if 'model' in locals():
            user_input = np.array([[clump_thickness, cell_size, cell_shape, marginal_adhesion,
                                   single_epithelial_cell_size, bare_nuclei, bland_chromatin,
                                   normal_nucleoli, mitoses]])
            columns = ["Clump_Thickness", "Uniformity_of_Cell_Size", "Uniformity_of_Cell_Shape",
                       "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei",
                       "Bland_Chromatin", "Normal_Nucleoli", "Mitoses"]
            input_df = pd.DataFrame(user_input, columns=columns)
            prediction = model.predict(input_df)

            st.markdown("### <div style='text-align: center;'>النتيجة:</div>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.error("🔴 النتيجة: خبيث (Malignant)")
            else:
                st.success("🟢 النتيجة: حميد (Benign)")
            
            st.markdown("---")
            st.write("البيانات المدخلة:")
            st.dataframe(input_df, use_container_width=True)

            st.markdown("---")
            st.subheader("تصور حدود قرار النموذج")
            
            data_df = pd.read_csv("breast-cancer-wisconsin.data", names=column_names, na_values="?")
            data_df.dropna(inplace=True)
            data_df['Class'] = data_df['Class'].map({2: 0, 4: 1})
            data_df['Bare_Nuclei'] = pd.to_numeric(data_df['Bare_Nuclei'])
            
            X_full = data_df.drop(['Sample_code_number', 'Class'], axis=1)
            y_full = data_df['Class']

            fig = plot_decision_boundary(model, X_full, y_full)
            st.pyplot(fig)
            
# --- واجهة معلومات الخصائص (الجديدة والمحسّنة) ---
elif page == "معلومات عن الخصائص":
    st.markdown("<h1 style='text-align: center;'>معلومات تفصيلية عن خصائص العينة</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.write("هذه الصفحة توضح معاني كل خاصية من الخصائص التسعة المستخدمة في نموذج التنبؤ، بالإضافة إلى تأثير كل منها على قرار النموذج.")
    
    st.markdown("---")
    
    for key, info in feature_info.items():
        st.markdown(f"""
            <div style='
                background-color: #F8F9FA; 
                border-left: 5px solid #3366ff; 
                border-radius: 8px; 
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            '>
                <h3 style='color: #3366ff; margin-top: 0;'>{info['arabic_name']} ({key})</h3>
                <p style='font-size: 16px; color: #495057; font-weight: bold;'>تعريف:</p>
                <p style='font-size: 16px; color: #495057;'>{info['description']}</p>
                <br>
                <p style='font-size: 16px; color: #495057; font-weight: bold;'>التأثير على القرار:</p>
                <p style='font-size: 16px; color: #495057;'>{info['impact']}</p>
            </div>
            """, unsafe_allow_html=True)