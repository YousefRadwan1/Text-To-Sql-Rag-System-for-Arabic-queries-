import streamlit as st
import os
from rag_system import RAGSystem  # Importing the RAGSystem class

# Initialize the Gemini API key (you can use environment variables or hardcode it for testing)
gemini_api_key = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyBqSqj418HluZowR58hDrmwOmnLf_7x7cA"
)

# Initialize RAG system with embedding, vector DB, and LLM
rag_system = RAGSystem(gemini_api_key=gemini_api_key)

# Load or build the index at app start
try:
    rag_system.build_index("AR_spider.jsonl", force_rebuild=False)
except Exception as e:
    st.warning(f"Warning: Could not load or build index: {e}")

# Streamlit Header and Title
st.title("AR-spider Databases SQL Query Generator")

# Streamlit Form for user input
with st.form("query_form"):
    question = st.text_area("Ask your database question...", height=150)
    
    # Removed db_id and top_k input fields
    submit_button = st.form_submit_button("Generate SQL")

# Default values for db_id and top_k
default_db_id = "" 
""  # Set a default database ID, change as needed
default_top_k = 5  # Set the default number of results to return

# Generate SQL query when form is submitted
if submit_button:
    if question:
        with st.spinner("Generating SQL query..."):
            try:
                # Generate SQL query using RAGSystem, use default values for db_id and top_k
                sql_query = rag_system.query(question=question, db_id=default_db_id, top_k=default_top_k)
                st.success("SQL query generated successfully!")
                st.code(sql_query, language="sql")
            except Exception as e:
                st.error(f"Error generating SQL: {str(e)}")
    else:
        st.warning("Please enter a question to generate the SQL query.")

# Optional: Sidebar for Rebuilding Index (if you want that functionality)
st.sidebar.header("Rebuild Index")
with st.sidebar.form("rebuild_form"):
    rebuild_button = st.form_submit_button("Rebuild Index")
    if rebuild_button:
        try:
            rag_system.build_index("AR_spider.jsonl", force_rebuild=True)
            st.sidebar.success("Index rebuilt successfully!")
        except Exception as e:
            st.sidebar.error(f"Error rebuilding index: {str(e)}")
