from streamlit_interface import st
import app_utils as a_u


def playground_tab():
    """
    This function provides an interface for the Playground tab in the application.
    After the user clicks the "Apply Configuration" button, the function updates the session state with the selected settings and parameters.
    """
    st.write("**Tuning**.🛝")
    st.write("Adjust the application settings and parameters to suite your use case.",
             "Don't forget to click the **Apply Configuration** button at the bottom after editing")
    st.write("⚠️: If you end up getting errors, readjust the parameters or click the Reset Buttons!")

    with st.container(border=True):   
        
        # Agent Selection
        with st.expander("Agent selection"):
            agent_selection = st.selectbox("Agent list", options=["👶 simple","🧑‍🎓 academic","🧑‍🔧 technical","🧑‍🏫 didactic","🤖 concise"])
                
        # Vector Storage Selection
        with st.expander("Vector Storage"):
            if st.toggle("Use Online Vector Storage"):
                vector_selection = st.selectbox("Select Online Vector Storage", options=["Pinecone"])
                if vector_selection == "Pinecone":
                    st.session_state.pinecone_api_key = st.text_input("Pinecone API Key", type="password")
                    st.session_state.pinecone_index = st.text_input("Pinecone Index")
                    st.write("⚠️: The index records will be cleared and started afresh")
            else:
                vector_selection = st.selectbox("Select Local Vector Storage", options=["FAISS","chromadb"])
                
        # Embedding Model Selection        
        with st.expander("Embedding Model"):
            embedding_model = st.selectbox("Select Embedding Model", options=["HuggingFaceEmbeddings"],
                                            help="Select the embedding model to use for the application")

        with st.container(border=True):
            st.write("**Adjust Parameters** ")
            
            # Model Parameters  
            with st.expander("Model"):
                model_temperature = st.slider("temperature", key="slider_model_temperature", min_value=0.0,
                                              max_value=1.0, step=0.1, value=st.session_state.model_temperature,
                                              help="Temperature controls the randomness or creativity of the generated text")
                st.button("Reset", on_click=a_u.reset_slider_value, args=(a_u.model_reset_dict,), key="model_param_reset")
                
            # Text Splitter Parameters  
            with st.expander("Text Splitter"):
                max_token = a_u.model_max_context_limit[st.session_state.get("endpoint").split("@")[0]]
                chunk_size = st.slider("chunk_size", key="slider_chunk_size", min_value=200, max_value=max_token, step=100, # max_token given by model_max_context_limit
                                       value=st.session_state.chunk_size,
                                       help="The maximum size of each chunk in tokens")
                max_overlap = min(chunk_size - 99, 1000)    # chunk_size - 99 to avoid overlap > chunk_size
                chunk_overlap = st.slider("Chunk Overlap", key="slider_chunk_overlap", min_value=0,
                                          max_value=max_overlap, step=100, value=st.session_state.chunk_overlap,
                                          help="The number of tokens to overlap between chunks")
                st.button("Reset", on_click=a_u.reset_slider_value, args=(a_u.splitter_reset_dict,),
                          key="text_splitter_param_reset")
                
            # Retriever Parameters
            with st.expander("Retirever"):
                search_type = st.selectbox("Search Type", options=["similarity", "mmr", "similarity_score_threshold"],
                                           help="Defines the type of search that the Retriever should perform")
                k = st.slider(
                    "k",
                    key="slider_k",
                    help="Amount of documents to return (Default: 4)",
                    min_value=1,
                    max_value=100,
                    value=st.session_state.k
                )
                
                if search_type == "similarity_score_threshold":
                    score_threshold = st.slider(
                        "score_threshold",
                        key="slider_score_threshold",
                        help="Minimum relevance threshold for a document to be returned",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        value=st.session_state.score_threshold,
                    )

                if search_type == "mmr":
                    fetch_k = st.slider(
                        "fetch_k",
                        key="slider_fetch_k",
                        help="Amount of documents to pass to MMR algorithm",
                        value=st.session_state.fetch_k
                    )
                    lambda_mult = st.slider(
                        "lambda_mult",
                        key="slider_lambda_mult",
                        help="Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.1,
                        value=st.session_state.lambda_mult
                    )

                # TODO
                # with st.container(border=True):
                #     st.markdown("filter", help="Filter by document metadata")
                #
                #     col1, col2 = st.columns([0.5, 1])
                #     with col1:
                #         st.text_input(label="key")
                #
                #     with col2:
                #         st.text_input(label="value")
                #
                # with st.container(border=True):
                #     st.markdown("Set Max Tokens", help="The retrieved document tokens will be checked and reduced below this limit.")
                #     st.slider("max_tokens_retrieved")

                st.button("Reset", on_click=a_u.reset_slider_value, args=(a_u.retriever_reset_dict,),
                          key="retriever_param_reset")

            st.session_state.applied_config = False

        # History Unaware Toggle
        with st.expander("Extras"):
            with st.container(border=True):
                st.markdown("**History Unaware**")
                st.write("Useful when playing around with parameters (Input Cost Friendly)")

                if st.toggle("History Unaware", value=st.session_state.history_unaware, help="Enable for a simple Q&A app with no history attached"):
                    history_unaware = True
                else:
                    history_unaware = False
                    
        # Apply Configuration Button
        if st.button("Apply Configuration", on_click=a_u.field_callback, args=("Configuration",), key="apply_params_config",
                     type="primary"):
            st.session_state.embedding_model = embedding_model
            st.session_state.vector_selection = vector_selection
            st.session_state.agent_selection = agent_selection
            st.session_state.model_temperature = model_temperature
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.history_unaware = history_unaware

            if st.session_state.history_unaware:
                st.session_state.messages = []

            st.session_state.search_type = search_type
            st.session_state.k = k

            if search_type == "similarity_score_threshold":
                st.session_state.score_threshold = score_threshold

            if search_type == "mmr":
                st.session_state.fetch_k = fetch_k
                st.session_state.lambda_mult = lambda_mult

            st.session_state.applied_config = True

        # Process Documents outside Column
        if st.session_state.applied_config:
            a_u.process_inputs()
            st.session_state.applied_config = False

    # Clear Chat History Button
    if len(st.session_state.messages) > 0:
        st.button("🧹 Clear Chat History", key="play_clear_history", on_click=a_u.clear_history)
