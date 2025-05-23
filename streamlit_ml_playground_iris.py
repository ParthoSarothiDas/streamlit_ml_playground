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
    This section provides essential information about the Iris dataset, including its shape, unique values, summary statistics, 
    missing or duplicate entries, and the distribution of the target variable. Use the checkboxes below to explore different aspects of the dataset.
    """)
    st.markdown("---")

    if st.checkbox("Show dataset's first & last rows"):
        st.subheader("First five rows")
        st.write(df.head())
        st.subheader("Last five rows")
        st.write(df.tail())

    if st.checkbox("Show dataset shape"):
        st.write(f"Number of rows: **{df.shape[0]}**")
        st.write(f"Number of columns: **{df.shape[1]}**")
    if st.checkbox("Show unique values"):
        st.write(df.nunique())
    if st.checkbox("Show Statistics"):
        st.subheader("Numerical Columns")
        st.write(df.describe())
        st.subheader("Categorical Columns")
        st.write(df.describe(include=object))
    if st.checkbox("Missing and Duplicate values"):
        st.subheader("Missing values:")
        st.write(df.isnull().sum())
        st.subheader("Duplicated values:")
        st.write(df[df.duplicated()])
    if st.checkbox("Target Variable"):
        st.subheader("Class distribution of target variable")
        col1, col2, = st.columns(2)
        col1.text("Distriburion on counts:")
        col1.write(df[df.columns[-1]].value_counts())

        col2.text("Distribution on percentage:")
        col2.write(df[df.columns[-1]].value_counts(normalize=True)*100)
    
# Tab 3: Visualizations
with tab3:
    st.subheader("Exploratory Data Visualizations")
    st.markdown("""
    This section provides interactive visualizations to help uncover patterns, relationships, and distributions in the Iris dataset. 
    You can explore the features through boxplots, heatmaps, pairplots, histograms, and violin plots.
    """)
    st.markdown("---")
    feature_options = ["Select a feature"] + df.columns[:-1].tolist()
    selected_feature = st.selectbox("Select a feature for boxplot", feature_options)
    if selected_feature != "Select a feature":
        fig, ax = plt.subplots(figsize=(4,1))
        sns.boxplot(x=df[selected_feature], ax=ax, color='skyblue')
        ax.set_title(f"Boxplot of {selected_feature}")
        st.pyplot(fig)
    else:
        pass

  
    # Radio Buttons for Graph Selection
    status = st.radio("Select a Graph", options=['None','Heatmap','Pairplot','Histogram','Violin'], horizontal=True)
    # if status != 'None':
    if status == 'Heatmap':
        fig = plt.figure(figsize=(6, 2))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

    elif status == 'Pairplot':
        plt.figure()
        fig=sns.pairplot(data=df, hue='species', height=2)
        st.pyplot(fig)

    elif status == 'Histogram':
        fig = plt.figure(figsize=(8,4))
        for i, col in enumerate(df.columns[0:4]): # (numeric_df.columns):
            plt.subplot(2,2,i+1)
            sns.histplot(x=df[col], kde = True, color='lightblue')
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        st.pyplot(fig)

    elif status == 'Violin':
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,4))
        sns.violinplot(data=df, y='sepal_length', ax=ax[0,0])
        sns.violinplot(data=df, y='sepal_width', ax=ax[0,1])
        sns.violinplot(data=df, y='petal_length', ax=ax[1,0])
        sns.violinplot(data=df, y='petal_width', ax=ax[1,1])
        plt.tight_layout()
        st.pyplot(fig)

    else:
        pass

# tab 4: ML Models
with tab4:
    st.subheader("Train and Evaluate Machine Learning Models")
    st.markdown("""
    In this section, you can train various classification models on the Iris dataset and evaluate their performance. 
    Select a model to view its accuracy, training and test scores, and confusion matrix. This helps compare model effectiveness for species prediction.
    """)
    st.markdown("---")

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
            "Model Name":[model_name],
            "Accuracy Score":[accuracy],
            "Training Score":[training_score],
            "Test Score": [test_score]
            })
        
        st.write("Accuracy Table",df_result)
        
        col1, col2 = st.columns([2,1])
        report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
        st.markdown('---')
        col1.write("Classification Report")
        col1.dataframe(pd.DataFrame(report).transpose())
        col2.write("Confusion Matrix")
        col2.write(cm)
        return

    if st.checkbox("AdaBoostClassifier"):
        ml_classifier(AdaBoostClassifier())

    if st.checkbox("KNeighborsClassifier"):
        ml_classifier(KNeighborsClassifier())

    if st.checkbox("DecisionTreeClassifier"):
        ml_classifier(DecisionTreeClassifier())
    
    if st.checkbox("RandomForestClassifier"):
        ml_classifier(RandomForestClassifier())
    
    if st.checkbox("XGBClassifier"):
        ml_classifier(XGBClassifier())
    
    if st.checkbox("SVC"):
        ml_classifier(SVC())


# tab 5 : Prediction
with tab5:
    st.subheader("Predict Iris Species from Flower Measurements")
    st.markdown("""
    Use this section to **predict the species of an Iris flower** based on user-provided feature inputs. 
    Enter values for sepal and petal dimensions, and choose a classification model to see the predicted species along with the prediction probabilities.  
    This is a hands-on way to see how machine learning models generalize to new, unseen data.
    """)
    st.markdown("---")

    st.markdown('**User Input Parameters**')

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

    df_user = user_input_features()
    # st.write("**User Input Parameters**")
    # st.write(df_user)

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
        st.markdown("---")
        return

    if st.checkbox("AdaBoostClassifier", key=1):
        ml_predictor(AdaBoostClassifier())

    if st.checkbox("KNeighborsClassifier", key=2):
        ml_predictor(KNeighborsClassifier())

    if st.checkbox("DecisionTreeClassifier", key=3):
        ml_predictor(DecisionTreeClassifier())

    if st.checkbox("RandomForestClassifier", key=4):
        ml_predictor(RandomForestClassifier())

    if st.checkbox("XGBClassifier", key=5):
        ml_predictor(XGBClassifier())

    if st.checkbox("SVC", key=6):
        ml_predictor(SVC(probability=True))
