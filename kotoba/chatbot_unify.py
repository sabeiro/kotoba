from unify import Unify
import re

templateS = """GENERAL INSTRUCTION
Your task is to answer questions, if you cannot answer the question request a helper or use a tool. Fill with Nil where no tool or helper is required.

AVAILABLE TOOLS
- Search tool
- Math tool

AVAILABLE HELPERS
- Decomposition: breaks complex questions into simpler subparts

CONTEXTUAL INFORMATION
{context}

QUESTION
{question}

ANSWER FORMAT
"Answer":"<Fill>"
"""

yesNo = re.compile(r'^\s*(yes|no).*',flags=re.IGNORECASE)
yesRe = re.compile(r'^\s*(yes).*',flags=re.IGNORECASE)

def get_unify(modelN="mistral-7b-instruct-v0.2@fireworks-ai",temp=.3):
    model = Unify(modelN)
    #model = ChatUnify(model=modelN,unify_api_key=os.environ['UNIFY_KEY'],temperature=temp)
    return model

def ask_rag(q,retS,unify):
    #prompt = f"Read this text and answer the question: {q}:\n{doc}"
    prompt = templateS.format(context = retS, question = q)
    ans = unify.generate(prompt)
    return ans

def ask_question(q,retL,unify):
    #prompt = f"Read this text and answer the question: {q}:\n{doc}"
    prompt = "The following document answers "+q+":\n\n{doc} \n\n Answer your confidence"
    ansL = []
    for doc in retL:
        ans = unify.generate(prompt)
        ansL.append(ans)
    return ansL

def rank_question(resL,unify):
    doc = ".".join([str(i) + ") " + x for i,x in enumerate(resL)])
    prompt = f"Which of the following answer is confident in the following document?:\n\n{doc}\n\n please mention numbers only"
    ans = unify.generate(prompt)
    return ans

def ask_unify(query="Two plus two is even? Respond with true/false"):
    unify = Unify("mistral-7b-instruct-v0.2@fireworks-ai")
    response = unify.generate(query)
    return response


if False:
    url = "https://api.unify.ai/v0/chat/completions"
    headers = {"Authorization": "Bearer " + os.environ['UNIFY_KEY'],'Content-Type':'application/json'}
    payload = {"model": "mistral-7b-instruct-v0.2@fireworks-ai","messages": [{"role": "user",
                                                                              "content": "two plus two is?"}],"stream": True}
    response = requests.post(url, json=payload, headers=headers, stream=True)
    print(response.status_code)
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode("utf-8"))
                resp = json.loads(chunk.decode("utf-8")[6:])
    else:
        print(response.text)

