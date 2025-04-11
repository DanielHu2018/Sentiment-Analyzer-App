import streamlit as st

### Title/Header ###
st.title("Sentiment Analyzer by Daniel Hu")
st.header("Home", divider="red")

### Sidebar ###
home = st.sidebar.page_link("app.py", label="Home", icon = "🏠")
results = st.sidebar.page_link("pages/results.py", label="Results", icon = "🚀")
code = st.sidebar.page_link("pages/codebase.py", label = "Codebase", icon = "🤖")

st.write("Welcome!")
st.write("Continue exploring using the sidebar.")