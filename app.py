import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
os.environ["PHI_API_KEY"] = os.getenv("PHI_API_KEY")

st.title("🧠 Financial Chatbot Test")

# Test Finance Agent
try:
    finance_agent = Agent(
        name='Finance Agent',
        model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
        tools=[YFinanceTools(stock_price=True)],
        instructions=["Provide stock data in a clear format."],
        show_tool_calls=True,
        markdown=True
    )
except Exception as e:
    st.error(f"Agent Initialization Error: {e}")
    st.stop()

# User Query
user_query = st.text_input("Ask about stock prices:")

if user_query:
    with st.spinner("Fetching response..."):
        try:
            response = finance_agent.run(user_query)
            if response and response.content:
                st.markdown(response.content)
            else:
                st.error("No response from the agent.")
        except Exception as e:
            st.error(f"Error: {e}")
