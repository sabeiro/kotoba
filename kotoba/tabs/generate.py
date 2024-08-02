from streamlit_interface import st
import app_utils as c_t
from pathlib import Path

@st.experimental_dialog("Source Code", width="large")
def generate_src():
    """Generates the source code for the selected embedding model and vector selection."""
    st.write("Get the requirements from the requirements.txt of the repository")
    st.link_button("Go to requirements",
                   "https://github.com/sabeiro/kotoba", type="primary")
    
    code = None
    file_path = None
    base_path = Path(__file__).parent
    
    if st.session_state["embedding_model"] == "HuggingFaceEmbeddings":
        if st.session_state["vector_selection"] == "FAISS":
            code_path = "../data/faiss_huggingface.py"
            file_path = (base_path / code_path).resolve()
        elif st.session_state["vector_selection"] == "Pinecone":
            code_path = "../data/pinecone_huggingface.py"
            file_path = (base_path / code_path).resolve()
                
    with (open(file_path, "r") as f):
        code = f.readlines()
        code = "".join(code).replace(
            'enter_endpoint', str(st.session_state.endpoint)
        ).replace(
            'enter_model_temperature', str(st.session_state.model_temperature)
        ).replace(
            'enter_chunk_size', str(st.session_state.chunk_size)
        ).replace(
            'enter_chunk_overlap', str(st.session_state.chunk_overlap)
        ).replace(
            'enter_search_type', str(st.session_state.search_type)
        ).replace(
            'enter_search_kwargs', str(st.session_state.search_kwargs)
        )
    st.code(code, language='python')


def generate_code_tab():
    """ 
    displays the current configuration of the application, including the endpoint, model parameters, 
    text splitter parameters, and retriever parameters. 
    And provides a button to generate the source code based on the current configuration.
    """
    
    st.write("Finished adjusting the parameters to fit your use case? Get your code here.")
    
    # Display the parameters
    with st.container(border=True):
        st.write("**Parameters**")
        with st.container(border=True):
            st.write("**Endpoint**: ")
            st.text("model: " + str(st.session_state.endpoint.split("@")[0]))
            st.text("provider: " + str(st.session_state.endpoint.split("@")[1]))
            st.text("temperature: " + str(st.session_state.model_temperature))

        with st.container(border=True):
            st.write("**Text Splitter**")
            st.text("chunk_size: " + str(st.session_state.chunk_size))
            st.text("chunk_overlap: " + str(st.session_state.chunk_overlap))

        with st.container(border=True):
            st.write("**Retriever**")
            st.write("*Vector store*: ", st.session_state.vector_selection)
            st.write("*Embedding Model*: ", st.session_state.embedding_model)
            st.write("*Retriever Keywords*: {")
            st.text("search_type: " + str(st.session_state.search_type))
            st.text("k: " + str(st.session_state.k))

            if st.session_state.search_type == "similarity_score_threshold":
                st.text("similarity_score_threshold: " + str(st.session_state.score_threshold))

            if st.session_state.search_type == "mmr":
                st.text("fetch_k: " + str(st.session_state.fetch_k))
                st.text("lambda_mult: " + str(st.session_state.lambda_mult))

            st.write("}")
            
    # Button to generate the source code
    if st.button("Generate Source Code", type="primary"):
        generate_src()
