import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import os
import io
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("❌ OpenAI API Key not found. Please check your .env file.")

# Streamlit UI
st.title("📊 AI-Powered Data Analysis Tool 🚀")
st.subheader("Unleash Insights from Your Data with AI!")

st.sidebar.header("🔍 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Recognize voice
def recognize_speech():
    recognizer = sr.Recognizer()
    language_map = {
        "English": "en-US",
        "Spanish": "es-ES",
        "French": "fr-FR",
        "German": "de-DE",
        "Hindi": "hi-IN",
        "Chinese (Mandarin)": "zh-CN",
        "Japanese": "ja-JP",
        "Arabic": "ar-SA"
    }
    selected_language = st.sidebar.selectbox("🌍 Choose Language", list(language_map.keys()), index=0)
    selected_language_code = language_map[selected_language]
    st.sidebar.write(f"✅ Selected Language: *{selected_language}*")
    with sr.Microphone() as source:
        st.info(f"🎤 Speak now in {selected_language}...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5)
            recognized_text = recognizer.recognize_google(audio, language=selected_language_code)
            st.success(f"🎤 You said: {recognized_text}")
            return recognized_text
        except sr.UnknownValueError:
            st.error("⚠ Could not understand. Try again.")
        except sr.RequestError as e:
            st.error(f"⚠ Speech recognition error: {e}")
    return None

# For chart download
def get_chart_download_link(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# AI prompt wrapper
def ask_openai(df, user_question):
    try:
        context = f"Data:\n{df.head(10).to_string(index=False)}"
        prompt = f"{context}\n\nQuestion: {user_question}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error from OpenAI: {e}"

# Main App Logic
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file, engine="openpyxl")
        if df.empty:
            st.error("🚨 The uploaded file is empty.")
        else:
            st.write("### 📂 Dataset Preview")
            st.dataframe(df.head(100))

            st.write("### 📈 Summary Statistics")
            st.write(df.describe())

            st.write("### ❗ Missing Values")
            missing_values = df.isnull().sum()
            st.write(missing_values)

            if missing_values.any():
                st.write("### 💡 Fill Missing Values")
                for column in df.columns[df.isnull().any()]:
                    st.write(f"#### {column} (Type: {df[column].dtype})")
                    if df[column].dtype in ['float64', 'int64']:
                        fill_na_option = st.radio(
                            f"Fill '{column}'?", ["Mean", "Median", "Mode", "Leave As Is"],
                            key=column)
                        if fill_na_option == "Mean":
                            df[column].fillna(df[column].mean(), inplace=True)
                        elif fill_na_option == "Median":
                            df[column].fillna(df[column].median(), inplace=True)
                        elif fill_na_option == "Mode":
                            df[column].fillna(df[column].mode()[0], inplace=True)
                    else:
                        fill_na_option = st.radio(
                            f"Fill '{column}'? (Categorical)", ["Mode", "Leave As Is"],
                            key=column)
                        if fill_na_option == "Mode":
                            df[column].fillna(df[column].mode()[0], inplace=True)

                if st.button("Show Cleaned Dataset and Summary"):
                    st.write("### 📊 Updated Summary")
                    st.write(df.describe())
                    st.write("### ✅ Missing Values After Cleaning")
                    st.write(df.isnull().sum())
                    st.sidebar.download_button("📥 Download Cleaned Dataset", df.to_csv(index=False), "cleaned_dataset.csv", "text/csv")

            st.write("### 💬 Ask a Question")
            query = st.text_input("Enter your question about the data:")
            if query:
                st.write("#### 🤖 AI Response:")
                st.write(ask_openai(df, query))

            if st.button("🎤 Ask AI with Your Voice"):
                voice_query = recognize_speech()
                if voice_query:
                    st.write("#### 🤖 AI Response:")
                    st.write(ask_openai(df, voice_query))

            st.write("### 📊 Generate Visualization")
            chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Histogram", "Box", "Heatmap"])
            x_col = st.selectbox("X-axis", df.columns)
            y_col = st.selectbox("Y-axis", df.columns)

            if st.button("Generate Chart"):
                try:
                    fig = plt.figure(figsize=(10, 6))
                    if chart_type == "Bar":
                        sns.barplot(data=df, x=x_col, y=y_col)
                    elif chart_type == "Line":
                        sns.lineplot(data=df, x=x_col, y=y_col)
                    elif chart_type == "Scatter":
                        sns.scatterplot(data=df, x=x_col, y=y_col)
                    elif chart_type == "Pie":
                        df[y_col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
                    elif chart_type == "Histogram":
                        sns.histplot(data=df, x=x_col, bins=20, kde=True)
                    elif chart_type == "Box":
                        sns.boxplot(data=df, x=x_col, y=y_col)
                    elif chart_type == "Heatmap":
                        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
                    plt.title(f"{chart_type} Chart")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    buf = get_chart_download_link(fig)
                    st.sidebar.download_button("📥 Download Chart", buf, f"{chart_type}_{x_col}_{y_col}.png", "image/png")
                except Exception as e:
                    st.error(f"⚠ Error generating chart: {e}")
    except Exception as e:
        st.error(f"⚠ File read error: {e}")
else:
    st.info("📥 Please upload a dataset to begin.")
