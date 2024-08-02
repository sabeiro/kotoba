import os
from streamlit_interface import st
import app_utils as c_t


def home_tab():
    """
    This function sets up the home tab.
    It sets up the input for Unify API Key, model and provider selection and document uploader.
    """
    try:
        st.session_state.unify_api_key = os.environ['UNIFY_KEY']
    except:
        st.session_state.unify_api_key = st.text_input("Unify API Key*", type="password", on_change=c_t.field_callback,placeholder="Enter Unify API Key", args=("Unify Key ",))
    
    model_name = st.selectbox("Select Model",options=c_t.modL,index=0,on_change=c_t.field_callback,placeholder="Model", args=("Model",))
    provider_name = st.selectbox("Select a Provider", options=c_t.dynamic_provider,
                                 on_change=c_t.field_callback,placeholder="Provider", args=("Provider",))
    st.session_state.endpoint = f"{model_name}@{provider_name}"

    st.session_state.pdf_docs = st.file_uploader(label="Upload PDF Document(s)*",type="pdf",accept_multiple_files=True)

    if st.button("Submit Document(s)", type="primary"):
        c_t.process_inputs()

    if len(st.session_state.messages) > 0:
        st.button("🧹 Clear Chat History", key="home_clear_history", on_click=c_t.clear_history)
