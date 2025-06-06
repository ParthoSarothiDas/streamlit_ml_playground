import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

st.set_page_config('iris', '🌸', layout='centered')
df = sns.load_dataset("iris")

# Page title
st.subheader("Interactive EDA & Species Prediction on the Iris Dataset")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Dataset Info", "Visualizations", "ML Models", "Prediction"])

# Tab 1: Introduction
with tab1:
    st.subheader("Introduction")
    st.markdown("""
    <p style='text-align: justify;'>
    This application performs an exploratory data analysis (EDA) and predictive modeling on the well-known <b>Iris flower dataset</b>. 
    The Iris dataset is a multivariate dataset introduced by the British statistician Ronald Fisher in 1936. It contains morphological measurements 
    of iris flowers from three different species: <i>setosa</i>, <i>versicolor</i>, and <i>virginica</i>. Using this dataset, we can explore feature relationships 
    and build machine learning models to classify the flower species.
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.image("setosa.jpg")
    col1.caption("Iris: Setosa")
    col2.image("versicolor.jpg")
    col2.caption("Iris: Versicolor")
    col3.image("virginica.jpg")
    col3.caption("Iris: Virginica")

    st.subheader("Objectives of this App")
    st.markdown("""
    This interactive web app is designed to:

    - Provide a quick overview and basic statistics of the Iris dataset.
    - Offer feature-level visualizations using boxplots, violin plots, histograms, and pairplots.
    - Help identify correlations, patterns, and class distributions.
    - Enable user-driven input to predict flower species using a trained Random Forest classifier.
    - Show prediction probabilities for better interpretability.

    Whether you're a data science beginner or enthusiast, this app will help you better understand the Iris dataset through interactive EDA and classification.
    """)

    st.subheader("Dataset Description")
    st.markdown("""
    The **Iris dataset** contains **150 samples**, each representing a flower. Each sample includes **four numerical features** and **one target label** (species):

    | Feature Name     | Description                                   | Type       |
    |------------------|-----------------------------------------------|------------|
    | `sepal_length`   | Sepal length in centimeters                   | Float      |
    | `sepal_width`    | Sepal width in centimeters                    | Float      |
    | `petal_length`   | Petal length in centimeters                   | Float      |
    | `petal_width`    | Petal width in centimeters                    | Float      |
    | `species`        | Iris species (`setosa`, `versicolor`, `virginica`) | Categorical |

    Each species has 50 samples, making the dataset perfectly balanced.
    """)
    

# Tab 2: Dataset Info
with tab2:
    st.subheader("Dataset Overview and Summary Statistics")
    st.markdown("""
    <p style='text-align: justify;'>
    This section provides essential information about the Iris dataset, including its shape, unique values, summary statistics, 
    missing or duplicate entries, and the distribution of the target variable. Use the radio button below to explore different aspects of the dataset.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([0.7,2])
    options = [None,'First & Last Rows','Shape of Dataset','Unique Values','Statistics','Missing & Duplicates','Target Variable']
    selected_info_type = col1.radio("Select a Graph", options=options, key=110)
    with col2:
        with st.container(border=True):
            if selected_info_type == "First & Last Rows":
                st.subheader("First five rows")
                st.table(df.head())
                st.subheader("Last five rows")
                st.table(df.tail())

            elif selected_info_type == "Shape of Dataset":
                st.subheader("Shape of the Dataset")
                st.write(f"Number of rows: **{df.shape[0]}**")
                st.write(f"Number of columns: **{df.shape[1]}**")

            elif selected_info_type == "Unique Values":
                st.subheader("Unique Values")
                st.table(df.nunique())

            elif selected_info_type == "Statistics":
                st.subheader("Numerical Columns")
                st.table(df.describe())
                st.subheader("Categorical Columns")
                st.table(df.describe(include=object))

            elif selected_info_type == "Missing & Duplicates":
                st.subheader("Missing values:")
                st.table(df.isnull().sum())
                st.subheader("Duplicated values:")
                st.table(df[df.duplicated()])

            elif selected_info_type == "Target Variable":
                st.subheader("Class distribution of target variable")
                col1, col2, = st.columns(2)
                col1.write("Distriburion on counts:")
                col1.table(df[df.columns[-1]].value_counts())

                col2.write("Distribution on percentage:")
                col2.table(df[df.columns[-1]].value_counts(normalize=True)*100)
    
# Tab 3: Visualizations
with tab3:
    st.subheader("Exploratory Data Visualizations")
    st.markdown("""
    <p style='text-align: justify;'>
    This section provides interactive visualizations to help uncover patterns, relationships, and distributions in the Iris dataset. 
    You can explore the features through boxplots, heatmaps, pairplots, histograms, and violin plots.
    </p>
    """, unsafe_allow_html=True)
    # -----------------------> Boxplot------------
    with st.container(border=True):
        feature_options = ["Select a feature"] + df.columns[:-1].tolist()
        selected_feature = st.selectbox("Select a feature for boxplot", feature_options)
        if selected_feature != "Select a feature":
            col1,col2 = st.columns(2)
            
            fig, ax = plt.subplots(figsize=(1,1.7))
            sns.boxplot(y=df[selected_feature], ax=ax, color='skyblue')
            col1.pyplot(fig)

            # Data analysis for storytelling
            feature_data = df[selected_feature].dropna()
            q1 = feature_data.quantile(0.25)
            q3 = feature_data.quantile(0.75)
            median = feature_data.median()
            iqr = q3 - q1
            outliers = feature_data[(feature_data < (q1 - 1.5 * iqr)) | (feature_data > (q3 + 1.5 * iqr))]

            # Storytelling block
            col2.markdown("### 📖 Insight")
            col2.markdown(f"""
            - The **median {selected_feature.replace('_', ' ')}** is approximately **{median:.2f} cm**, representing the central tendency.
            - The **Interquartile Range (IQR)** is **{iqr:.2f} cm**, indicating {"a tight clustering" if iqr < 0.5 else "a wide spread"} among 50% of the samples.
            - There are **{len(outliers)} potential outlier(s)**, which may reflect unusual observations or measurement variations.
            - Understanding the distribution of **{selected_feature.replace('_', ' ')}** is crucial for distinguishing iris species and preparing features for classification models.
            """)
    with st.container(border=True):
        # Radio Buttons for Graph Selection
        status = st.radio("Select a Graph", options=['None','Heatmap','Pairplot','Histogram','Violin'], horizontal=True, key=100)
        if status == 'Heatmap':
            st.subheader("🔍 Feature Correlation Heatmap")

            # Compute correlation matrix
            corr_matrix = df.drop(columns='species').corr()

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            ax.set_title("Feature Correlation Heatmap (Iris Dataset)")
            st.pyplot(fig)

            # Storytelling
            st.markdown("### 📖 Insight")
            st.markdown("""
            - The heatmap shows the **pairwise correlations** between the four numerical features in the Iris dataset.
            - 🔸 **Petal length** and **petal width** are **very strongly correlated** (**correlation ≈ 0.96**). This suggests that one of them might be redundant in some models.
            - 🔸 **Sepal length** has a **moderate positive correlation** with petal length (**≈ 0.87**), indicating some shared pattern across species.
            - 🔸 **Sepal width** shows a **weak negative correlation** with the other features, especially petal width (**≈ -0.37**), which could help in species separation.
            - These insights are useful for:
            - Understanding feature importance
            - Selecting variables for dimensionality reduction (like PCA)
            - Designing interpretable classification models
            """)
        elif status == 'Pairplot':
            st.subheader("🔍 Feature Pairplot")
            plt.figure()
            fig=sns.pairplot(data=df, hue='species', height=2)
            st.pyplot(fig)
            st.markdown("### 📖 Insight")
            st.markdown("""
            The pairplot shows **scatter plots between each pair of numerical features**, color-coded by iris species, along with histograms on the diagonal.

            - **Setosa** is clearly **separable from the other two species** across most feature pairs — especially in plots involving **petal length** and **petal width**. This suggests that even simple models could effectively classify Setosa.
            - **Versicolor** and **Virginica** show **some overlap**, particularly in features like **sepal width** and **sepal length**, but they start to diverge in **petal dimensions**.
            - The diagonal histograms show that:
            - **Petal length and width** are **bimodal or trimodal**, strongly aligned with species separation.
            - **Sepal features** are more overlapping, making them less powerful as standalone features.
            - Overall, the pairplot highlights that:
            - **Petal features** are the most informative for species classification.
            - There's a strong **linear relationship** between petal length and petal width.
                        
            This visualization supports the idea that **feature combinations** matter, and can guide feature engineering or dimensionality reduction (e.g., PCA).
            """)

        elif status == 'Histogram':
            st.subheader("🔍 Feature Histogram's")
            fig = plt.figure(figsize=(8,4))
            for i, col in enumerate(df.columns[0:4]): # (numeric_df.columns):
                plt.subplot(2,2,i+1)
                sns.histplot(x=df[col], kde = True, color='lightblue')
                plt.title(f'Distribution of {col}')
            plt.tight_layout()
            st.pyplot(fig)

            # Storytelling
            st.markdown("### 📖 Insight")
            st.markdown("""
            The histograms above show the **distribution of each feature** in the Iris dataset, with smooth KDE (Kernel Density Estimation) curves overlayed.
            -  **Sepal Length** appears **roughly normally distributed**, with a slight right skew. Most flowers have a sepal length between 5 and 6 cm.
            -  **Sepal Width** shows a **slightly left-skewed** distribution, and it’s more **spread out** compared to other features. This may affect model performance if not standardized.
            -  **Petal Length** and **Petal Width** are **clearly bimodal**, indicating that different species have distinct petal measurements. These features are likely **highly discriminative** for classification.    
            
            Overall, **petal-based features** are more informative due to their **bimodal and wider range**, which makes them powerful for distinguishing species like *Setosa*, *Versicolor*, and *Virginica*.
            """)


        elif status == 'Violin':
            st.subheader("🔍 Feature Violin Plot")
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,4))
            sns.violinplot(data=df, y='sepal_length', ax=ax[0,0])
            sns.violinplot(data=df, y='sepal_width', ax=ax[0,1])
            sns.violinplot(data=df, y='petal_length', ax=ax[1,0])
            sns.violinplot(data=df, y='petal_width', ax=ax[1,1])
            plt.tight_layout()
            st.pyplot(fig)

            # Storytelling
            st.markdown("### 📖 Insight")
            st.markdown("""
            The violin plots above provide a deeper look at the **distribution, density, and variability** of each numerical feature in the Iris dataset.
            -  **Sepal Length** and **Sepal Width** both show fairly **symmetrical** distributions, but **sepal width** has **a wider spread**, indicating more variation across samples.
            -  **Petal Length** and **Petal Width** both show **distinct peaks**, suggesting **multiple modes** — which aligns with the idea that different iris species have clearly different petal sizes.
            - The **fatter sections of each violin** indicate where data is concentrated, while the **thinner ends** represent rare or extreme values. Notably, **petal-based features** have clear variations in density, hinting at their strong relevance for classification.

            Compared to boxplots or histograms, violin plots help us visualize the **shape** of the data more smoothly, making them ideal for exploratory pattern recognition.
            """)

        # else:
        #     pass

# tab 4: ML Models
with tab4:
    st.subheader("Train and Evaluate Machine Learning Models")
    st.markdown("""
    <p style='text-align: justify;'>           
    In this section, you can train various classification models on the Iris dataset and evaluate their performance. 
    Select a model to view its accuracy, training and test scores, and confusion matrix. This helps compare model effectiveness for species prediction.
    </p>
    """, unsafe_allow_html=True)
    # st.markdown("---")

    df.drop_duplicates(inplace=True)

    # Encode target labels (species)
    encoder = LabelEncoder()
    df['species'] = encoder.fit_transform(df['species'])

    # Separate features and target
    X = df.drop("species", axis=1)
    y = df["species"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Generalized ML classifier function
    def ml_classifier(model):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_name = f"{model.__class__.__name__}"
        accuracy = accuracy_score(y_test, y_pred)
        training_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        cm = confusion_matrix(y_test, y_pred)

        df_result = pd.DataFrame({
            # "Model Name":[model_name],
            "Accuracy Score":[accuracy],
            "Training Score":[training_score],
            "Test Score": [test_score]
            })
        
        
        
        col1, col2 = st.columns([2,1])
        report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
        # st.markdown('---')
        col1.write("Accuracy Score")
        col1.write(df_result)
        col2.write("Confusion Matrix")
        col2.write(cm)
        st.write("Classification Report")
        st.table(pd.DataFrame(report).transpose())
        return

    with st.container(border=True):
        # Radio Buttons for Graph Selection
        model_dict = {
                "Random Forest": RandomForestClassifier(),
                "Ada Boost": AdaBoostClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "K Neighbors": KNeighborsClassifier(),
                "XG Boost": XGBClassifier(),
                "SVC": SVC(probability=True)
                }
        # option_list = 
        col1, col2 = st.columns([.6,2])
        selected_model = col1.radio("Select a Graph", options=[None] + list(model_dict.keys()), key=101)
        with col2:
            if selected_model != None:
                model = model_dict.get(selected_model)
                ml_classifier(model)

        if selected_model != None:
            st.markdown("### 📖 Insight")
            st.markdown(f"""
            You've just evaluated the **{selected_model}** on the classic Iris dataset.

            - The **accuracy score** indicates how well the model is generalizing to **unseen test data**, while the **training score** shows performance on the data the model has already seen.
            - A large gap between training and test scores might indicate **overfitting** (high train, low test) or **underfitting** (both low).
            - The **confusion matrix** breaks down predictions into correct and incorrect counts for each class (`setosa`, `versicolor`, `virginica`), helping identify if the model struggles with any specific species.
            - The **classification report** gives detailed performance metrics:
            - **Precision**: Of the predicted class samples, how many were correct?
            - **Recall**: Of the actual class samples, how many were correctly predicted?
            - **F1-score**: Harmonic mean of precision and recall (a balanced metric).
            
            This section helps users **compare multiple models interactively** to find which classifier works best for this dataset.
            """)


# tab 5 : Prediction
with tab5:
    st.subheader("Predict Iris Species from Flower Measurements")
    st.markdown("""
    <p style='text-align: justify;'>
    Use this section to predict the species of an Iris flower based on user-provided feature inputs. 
    Enter values for sepal and petal dimensions, and choose a classification model to see the predicted species along with the prediction probabilities.  
    This is a hands-on way to see how machine learning models generalize to new, unseen data.
    </p>
    """, unsafe_allow_html=True)
    # st.markdown("---")

    st.markdown('**User Input Parameters:**')

    def user_input_features():
        col1, col2, col3 = st.columns([4,1,4])
        sepal_length = col1.slider('Sepal Length (cm)', 4.3, 7.9, 5.4)
        sepal_width = col1.slider('Sepal Width (cm)', 2.0, 4.4, 3.4)
        petal_length = col3.slider('Petal Length (cm)', 1.0, 6.9, 1.3)
        petal_width = col3.slider('Petal Width(cm)', 0.1, 2.5, 0.2)
        data = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        features = pd.DataFrame(data, index=[0])
        return features
    with st.container(border=True): 
        df_user = user_input_features()
    
    def ml_predictor(model):
        # Scale the user input the same way as training data
        df_user_scaled = scaler.transform(df_user)

        model.fit(X_train, y_train)
        prediction = model.predict(df_user_scaled)
        prediction_proba = model.predict_proba(df_user_scaled)

        # Decode species label
        predicted_species = encoder.inverse_transform([prediction[0]])[0]
        proba_df = pd.DataFrame(prediction_proba, columns=encoder.classes_)

        col1, col2 = st.columns(2)
        col1.write('Prediction')
        col1.subheader(f"`{predicted_species}`")
        col1.write('Prediction Probability')
        col1.write(proba_df)

        img = predicted_species + ".jpg"
        col2.image(img)
        # st.markdown("---")
        return

    if st.checkbox("AdaBoostClassifier", key=1):
        with st.container(border=True):
            ml_predictor(AdaBoostClassifier())

    if st.checkbox("KNeighborsClassifier", key=2):
        with st.container(border=True):
            ml_predictor(KNeighborsClassifier())

    if st.checkbox("DecisionTreeClassifier", key=3):
        with st.container(border=True):
            ml_predictor(DecisionTreeClassifier())

    if st.checkbox("RandomForestClassifier", key=4):
        with st.container(border=True):
            ml_predictor(RandomForestClassifier())

    if st.checkbox("XGBClassifier", key=5):
        with st.container(border=True):
            ml_predictor(XGBClassifier())

    if st.checkbox("SVC", key=6):
        with st.container(border=True):
            ml_predictor(SVC(probability=True))



# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<hr style="margin-top: 30px;">
<div style="text-align: center; font-size: 0.9em; color: gray;">
    Created by <b>Partho Sarothi Das</b><br>
    <i>Aspiring Data Scientist | Passionate about ML & Visualization</i><br>
    Email: <a href="mailto:partho52@gmail.com">partho52@gmail.com</a>
</div>
""", unsafe_allow_html=True)
