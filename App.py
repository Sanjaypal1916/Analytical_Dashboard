from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
import pandas as pd
import tempfile
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.agents import AgentExecutor

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please set the environment variable.")
    st.stop()


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="deepseek-r1-distill-llama-70b",
    temperature=0, 
    reasoning_format="parsed"
)



def get_agent(df):
    agent =  create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type= "tool-calling", 
        handle_parsing_errors=True, 
        allow_dangerous_code=True,
    )

    executor = AgentExecutor.from_agent_and_tools(
        agent=agent.agent,   # raw agent logic
        tools=agent.tools,   # tools (e.g. python repl)
        verbose=True,
        return_intermediate_steps=True
    )
    return executor



st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("📊 AI-Powered Analytical Dashboard")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
query = st.text_area("Ask a question about your data and click 'Generate'")

if uploaded_file:
    # try: 
        data = pd.read_csv(uploaded_file)
        st.session_state.df = data
        df = st.session_state.df
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        if st.button("generate"):
            agent = get_agent(df)

            with st.spinner("Thinking..."):
                response = agent.invoke(
                        {"input": query + " ?"},
                        config={
                            "configurable": {
                                "return_intermediate_steps": True
                            }
                        }
                    )
                st.subheader("AI Response:")
                st.markdown(f"```\n{response["output"]}\n```")

                for step in response["intermediate_steps"]:
                    tool_output = step[1]

                    if isinstance(tool_output, str):
                        # Strip backticks and markdown if present
                        if tool_output.startswith("```"):
                            tool_output = tool_output.replace("```", "").replace("csv", "").strip()

                        try:
                            df_result = pd.read_csv(io.StringIO(tool_output))
                            st.subheader("📄 DataFrame Output:")
                            st.dataframe(df_result)
                        except Exception:
                            st.text(tool_output)
                    else:
                        st.warning(f"⚠️ Unexpected tool output type: {type(tool_output)}")
                        st.text(str(tool_output))
