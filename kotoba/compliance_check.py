import os, sys, json, re
import pandas as pd
import numpy as np
import streamlit
import langchain as lc
#https://accessibility.hhs.texas.gov/docs/processes/EditingTagsToFixComplexTables.pdf

os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.chatbot_utils as c_t
import kotoba.chatbot_prompt as c_p
import importlib

modL = ["gpt-4o@openai","gpt-4-turbo@openai","gpt-3.5-turbo@openai","mixtral-8x7b-instruct-v0.1@aws-bedrock","llama-2-70b-chat@aws-bedrock","codellama-34b-instruct@together-ai","gemma-7b-it@fireworks-ai","claude-3-haiku@anthropic","claude-3-opus@anthropic","claude-3-sonnet@anthropic","mistral-7b-instruct-v0.1@fireworks-ai","mistral-7b-instruct-v0.2@fireworks-ai"]

importlib.reload(c_t)
importlib.reload(c_p)
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
audD = pd.read_csv(baseDir + 'audit.csv')
aud = audD.iloc[0]
q = aud['audit_question_en']
pdf_doc = baseDir + 'Policies.pdf'
#pdf_doc = baseDir + 'BaroneLamberto2.pdf'
collN = re.sub(".pdf","",pdf_doc).split("/")[-1]

if False:
    #docL = c_t.pdf_page([pdf_doc])
    docL = c_t.pdf2tree(pdf_doc)
    # docL = c_t.pdf2md(pdf_doc)
    collT, collS = c_t.create_collection(docL,collN,baseDir)
else:
    collT, collS = c_t.load_chroma(collN,baseDir)
    # index = c_t.load_faiss(pdf_doc,baseDir)
    # query_engine = index.as_query_engine()
    # response = query_engine.query(q)
    # print(response.response)
    # n = response.source_nodes[0]


if False:
    i = 0
    aud = audD.iloc[0]
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
            res = c_t.format_confidence(chain.invoke(q))
        except:
            continue
        res['question'] = q
        res["justification"] = aud['exp_reference_en']
        res['context'] = aud['Content of BAIT Chapter (all)']
        res['truth'] = aud['exp_result']
        resL.append(res)

    evalDf = pd.DataFrame(resL)
    evalDf.to_csv(baseDir + "pred_openai.csv",index=False)


#selL = collT.get(include=[],limit=5,offset=1)
unify = c_t.get_unify(modL[0])
db = c_t.get_vectorstore(baseDir,collN)
retL = db.similarity_search_with_relevance_scores(q)
#retL = collT.query(query_texts=[q],n_results=5)['documents'][0]
resL = c_t.ask_question(q,retL,unify)
rank = c_t.rank_question(resL,unify)
corrL = [resL[x] for x in eval(rank)]

print("te se qe te ve be te ne?")
