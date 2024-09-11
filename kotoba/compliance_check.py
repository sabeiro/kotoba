import os, sys, json, re
import pandas as pd
import numpy as np
#https://accessibility.hhs.texas.gov/docs/processes/EditingTagsToFixComplexTables.pdf

os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.chatbot_utils as c_t
import kotoba.app_utils as a_u
import kotoba.chatbot_unify as c_u
import kotoba.chatbot_prompt as c_p
import importlib
import requests

importlib.reload(c_t)
importlib.reload(c_p)
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
audD = pd.read_csv(baseDir + 'audit.csv')
aud = audD.iloc[0]
q = aud['audit_question_en']
baseDir = os.environ['HOME'] + '/lav/src/kotoba/data/'
pdf_doc = baseDir + 'Policies.pdf'
pdf_doc = baseDir + 'data_proc.pdf'
#pdf_doc = baseDir + 'BaroneLamberto2.pdf'
collN = re.sub(".pdf","",pdf_doc).split("/")[-1]

docL = c_t.pdf2md(pdf_doc)

importlib.reload(c_t)
import streamlit as st
docL = c_t.pdf_page([pdf_doc])
vector_store = c_t.faiss_vector_storage(docL,collN="web",baseDir=baseDir)
#vector_store = c_t.create_collection(docL,collN="web",baseDir=baseDir)
query = "what is a data process agreement?" 
st.chat_message("human").write(query)
retriever = vector_store.as_retriever()
model = c_t.get_llm()
get_history = c_t.get_chat_message()
#rag_engine = c_t.create_conversational_rag_chain(model, retriever, get_history)
rag_engine = c_t.create_qa_chain(model, retriever)
response = st.chat_message("assistant").write_stream(a_u.output_chunks(rag_engine, query))
#st.session_state.messages.append((query, response))


retriever.get_relevant_documents("what is a data processing agreement?")
retriever.vectorstore.similarity_search("what is a data processing agreement?") 



if False:
    #docL = c_t.pdf_page([pdf_doc])
    #docL = c_t.pdf2tree(pdf_doc)
    docL = c_t.pdf2md(pdf_doc)
    collT, collS = c_t.create_collection(docL,collN,baseDir)
else:
    collT, collS = c_t.load_chroma(collN,baseDir)
    # index = c_t.load_faiss(pdf_doc,baseDir)
    # query_engine = index.as_query_engine()
    # response = query_engine.query(q)
    # print(response.response)
    # n = response.source_nodes[0]

if False: #langchain
    importlib.reload(c_p)
    importlib.reload(c_t)
    llm = c_t.get_llm()
    chain = c_t.get_chain_confidence(llm,baseDir,collN)
    resL = []
    for i, aud in audD.iterrows():
        print("%0.2f" % (100.*i/audD.shape[0]),end="\r")
        q = aud['audit_question_en']
        if q == '' or q != q:
            continue
        try:
            ans = c_t.format_confidence(chain.invoke(q))
        except:
            continue
        res['question'] = q
        res['pred_answer'] = ans['answer']
        res['pred_justification'] = ans['confidence']
        res['pred_context'] = ''
        res["ref_justification"] = aud['exp_reference_en']
        res['ref_context'] = aud['Content of BAIT Chapter (all)']
        res['ref_answer'] = aud['exp_result']
        resL.append(res)

    evalDf = pd.DataFrame(resL)
    evalDf.to_csv(baseDir + "pred_" + modN + ".csv",index=False)


resp = requests.get('https://api.unify.ai/v0/models',headers={"Authorization":"Bearer " + os.environ['UNIFY_KEY']})
modL = resp.text
modL = ["gpt-4o@openai","gpt-3.5-turbo@openai","mixtral-8x7b-instruct-v0.1@aws-bedrock","claude-3-haiku@anthropic","claude-3-opus@anthropic","claude-3-sonnet@anthropic"]
#selL = collT.get(include=[],limit=5,offset=1)
db = c_t.get_vectorstore(baseDir,collN)
importlib.reload(c_u)
for j, m in enumerate(modL): # unify
    try:
        unify = c_u.get_unify(modL[j])
    except:
        continue
    modN = modL[j].split("@")[0]
    print(modN)
    resL = []
    for i, aud in audD.iterrows():
        print("%0.2f" % (100.*i/audD.shape[0]),end="\r")
        q = aud['audit_question_en']
        if q == '' or q != q:
            continue
        retL = db.similarity_search_with_relevance_scores(q)
        retS = "\n".join([x[0].metadata['s'] for x in retL])
        ansS = c_u.ask_rag(q,retS,unify)
        ansD = eval("{"+ans+"}")
        res = {}
        yes = False
        try:
            if re.search(c_u.yesRe,ansD['Answer'].split(",")[0]):
                yes = True
        except:
            if re.search(c_u.yesRe,ansS):
                yes = True
        res['pred_answer'] = yes
        res['pred_justification'] = ans
        res['pred_context'] = retS
        res['question'] = q
        res["ref_justification"] = aud['exp_reference_en']
        res['ref_context'] = aud['Content of BAIT Chapter (all)']
        res['ref_answer'] = aud['exp_result']
        resL.append(res)

    evalDf = pd.DataFrame(resL)
    evalDf.to_csv(baseDir + "pred_" + modN + ".csv",index=False)

        

print("te se qe te ve be te ne?")
