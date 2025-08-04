import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Generate Dashboard", layout="wide")
st.title("📊 Generate Analytical Dashboard")

# Button to trigger dashboard
if "df" not in st.session_state:
    st.warning("⚠️ No dataset uploaded from the main page.")
else:
    if st.button("Generate Analytical Dashboard"):
        df = st.session_state.df

        st.subheader("📁 Dataset Preview")
        st.dataframe(df)

        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns

        # Histograms
        st.subheader("📊 Histograms")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)

        # Bar charts for categoricals
        st.subheader("📊 Categorical Distributions")
        for col in categorical_cols:
            if df[col].nunique() < 20:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Count of each category in {col}')
                st.pyplot(fig)

        # Time series plots
        st.subheader("📈 Time Series (if available)")
        if len(datetime_cols) == 0:
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col])
                        datetime_cols = [col]
                        break
                    except:
                        pass

        for time_col in datetime_cols:
            for num_col in numeric_cols:
                fig, ax = plt.subplots()
                df_sorted = df.sort_values(time_col)
                ax.plot(df_sorted[time_col], df_sorted[num_col])
                ax.set_title(f'{num_col} over time ({time_col})')
                st.pyplot(fig)

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.subheader("🔗 Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
