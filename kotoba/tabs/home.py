import os
from streamlit_interface import st
import app_utils as a_u


def home_tab():
    """
    This function sets up the home tab.
    It sets up the input for Unify API Key, model and provider selection and document uploader.
    """
    try:
        st.session_state.unify_api_key = os.environ['UNIFY_KEY']
    except:
        st.session_state.unify_api_key = st.text_input("API Key*", type="password", on_change=a_u.field_callback,placeholder="Enter API Key", args=("Unify Key ",))
    
    model_name = st.selectbox("Select Model",options=a_u.modL,index=0,on_change=a_u.field_callback,placeholder="Model", args=("Model",))
    provider_name = st.selectbox("Select a Provider", options=a_u.dynamic_provider,
                                 on_change=a_u.field_callback,placeholder="Provider", args=("Provider",))
    st.session_state.endpoint = f"{model_name}@{provider_name}"

    st.session_state.pdf_docs = st.file_uploader(label="Upload PDF Document(s)*",type="pdf",accept_multiple_files=True)

    if st.button("Submit Document(s)", type="primary"):
        a_u.process_inputs()

    if len(st.session_state.messages) > 0:
        st.button("🧹 Clear Chat History", key="home_clear_history", on_click=a_u.clear_history)
        st.button("📚 Cite response", key="home_cite", on_click=a_u.cite_response)
