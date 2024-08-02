import streamlit as st
import app_utils as a_t 
from tabs.home import home_tab
from tabs.play import playground_tab
from tabs.generate import generate_code_tab

def session_add(key, value, is_func=False):
    """
    Adds a key-value pair to the session state.
    
    Args:
        - key (str): The key to add to the session state.
        - value (str): The value to add to the session state.
        - is_func (bool): If True, calls the function `value` and adds the result to the session state.
    """
    if key not in st.session_state:
        if is_func:
            st.session_state[key] = value()
        else:
            st.session_state[key] = value


def init_keys():
    """Initializes session keys."""
    # All new session variables should be added here.
    session_add("chroma_persisted", False)
    session_add("vector_selection", "FAISS")
    session_add("embedding_model", "HuggingFaceEmbeddings")
    session_add("chunk_size", 1000)
    session_add("chunk_overlap", 100)
    session_add("messages", [])
    session_add("model_temperature", 0.3)
    session_add("store", {})
    session_add("search_type", "similarity")
    session_add("k", 4)
    session_add("fetch_k", 20)
    session_add("lambda_mult", 0.5)
    session_add("score_threshold", 0.5)
    session_add("history_unaware", False)
    session_add("search_kwargs", {})

def render_site():
    """Configures and displays the landing page."""
    st.set_page_config("Document checker", page_icon="👁️‍🗨️")
    st.title("Knowledge base LLM 💬")
    st.text("Chat with your PDF file using the LLM of your choice")
    st.write('''
            Usage: 
            1. export or define your UNIFY_KEY 
            2. Select the **Model** and endpoint provider of your choice from the drop down.
            3. Upload your document(s) and click the Submit button
            4. Chat
            ''')

    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('assistant').write(message[1])
        
    with st.sidebar:
        tab1, tab2, tab3 = st.tabs(["🏠Home", "🕹️Tuning", "👾Code"])
        with tab1:
            home_tab()
        with tab2:
            playground_tab()
        with tab3:
            generate_code_tab()
            
    a_t.chat_bot()


def main():
    st.set_page_config(page_title="audit compliance check",page_icon=":books:")
    st.header("metric comparison")
    st.text_input("ask a question")
    with st.sidebar:
        st.subheader("read doc")
        st.file_uploader("upload pdf")
        

if __name__ == '__main__':
    init_keys()
    render_site()
