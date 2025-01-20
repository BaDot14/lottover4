import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Title of the app
st.title("Lotto 4D Predictor with Enhanced Features")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your Lotto 4D file (CSV format only)", type=["csv", "txt"])

if uploaded_file:
    try:
        # Attempt to read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            data = pd.read_csv(uploaded_file, delimiter="\t")
        else:
            st.error("Unsupported file type. Please upload a CSV or TXT file.")
            data = None

        if data is not None:
            st.write("### Dataset Preview")
            st.write(data.head())

            # Step 2: Handle Missing or Null Values
            if data.isnull().values.any():
                st.write("### Handling Missing Values")
                data = data.fillna(method='ffill').fillna(method='bfill')
                st.write("Missing values handled by forward and backward filling.")

            # Rename and preprocess columns
            data.columns = ['Index', 'Date', 'Draw1', 'Draw2', 'Draw3', 'Draw4']
            data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
            data = data.drop(columns=['Index'])

            # Display cleaned dataset
            st.write("### Cleaned Dataset")
            st.write(data.head())

            # Step 3: Feature Engineering
            st.write("### Feature Engineering")
            scaler = StandardScaler()
            X = data[['Draw1', 'Draw2', 'Draw3', 'Draw4']].values
            X_scaled = scaler.fit_transform(X)
            st.write("Data scaled using StandardScaler.")

            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X_scaled)
            st.write("Polynomial features created with degree 2.")

            y = data['Draw1'].shift(-1).fillna(0).astype(int).values

            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

            # Step 4: Data Analysis and Visualization
            st.write("### Data Analysis and Visualization")
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            sns.histplot(data['Draw1'], kde=True, ax=axes[0, 0], color='blue').set(title='Distribution of Draw1')
            sns.boxplot(data=data[['Draw1', 'Draw2', 'Draw3', 'Draw4']], ax=axes[0, 1]).set(title='Boxplot of Draws')
            sns.heatmap(data[['Draw1', 'Draw2', 'Draw3', 'Draw4']].corr(), annot=True, cmap="coolwarm", ax=axes[1, 0]).set(title='Correlation Heatmap')
            sns.pairplot(data[['Draw1', 'Draw2', 'Draw3', 'Draw4']]).fig.suptitle('Pair Plot', y=1.02)
            st.pyplot(fig)

            # Step 5: Model Building
            st.write("### Model Building")

            models = {
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Logistic Regression': LogisticRegression(max_iter=500)
            }

            model_scores = {}

            for name, model in models.items():
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                model_scores[name] = scores.mean()
                st.write(f"{name} Cross-Validation Accuracy: {scores.mean() * 100:.2f}%")

            # Select the best model based on cross-validation
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            st.write(f"Best Model: {best_model_name}")
            st.write(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

            # Step 6: Evaluation and Visualization
            st.write("### Model Evaluation and Visualization")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            st.pyplot(fig)

            # Classification Report
            st.write("#### Classification Report")
            st.text(classification_report(y_test, y_pred))

            # ROC Curve
            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                ax.set_title('Receiver Operating Characteristic')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                st.pyplot(fig)

            # Step 7: Suggestions
            st.write("### Suggested Numbers")

            def suggest_numbers(data):
                all_numbers = data[['Draw1', 'Draw2', 'Draw3', 'Draw4']].values.flatten()
                number_counts = pd.Series(all_numbers).value_counts()
                top_numbers = number_counts.head(4).index.tolist()
                st.write("Most Frequent Numbers:", top_numbers)

            suggest_numbers(data)

            # Step 8: Download Predictions
            predictions_df = pd.DataFrame({'Predicted Numbers': y_pred})
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")
