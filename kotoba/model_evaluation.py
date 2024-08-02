import os, sys, json, re
import mlflow
import pandas as pd
import numpy as np
from mlflow.metrics import rougeLsum, rouge2
from mlflow.metrics.genai import EvaluationExample, faithfulness, answer_similarity, make_genai_metric, answer_relevance, answer_correctness, faithfulness
from mlflow.models import MetricThreshold, infer_signature
from langchain.evaluation.qa import QAEvalChain
#python3 -m mlflow server --host 0.0.0.0 --port 5151
#python3 -m mlflow gc

os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
pdf_doc = baseDir + 'Policies.pdf'
collN = re.sub(".pdf","",pdf_doc).split("/")[-1]
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.chatbot_utils as c_t
import importlib

ansDf = pd.read_csv(baseDir + "pred_openai.csv")
ansDf.loc[:,"pred_answer"] = ansDf['answer'].apply(lambda x: str(x))
ansDf.loc[:,"ref_answer"] = ansDf['truth'].apply(lambda x: str(x))
ansDf.loc[:,"inputs"] = ansDf['question'].apply(lambda x: str(x))
ansDf.dropna(subset=['pred_answer','ref_answer'],inplace=True)
ansMl = mlflow.data.from_pandas(ansDf, targets="truth", predictions="answer")
eval_data = ansDf[['reason','question','confidence','context','truth','answer']]
# ml_client = mlflow.MlflowClient(tracking_uri="http://0.0.0.0:5151")
# mlflow.set_tracking_uri("http://0.0.0.0:5151")
ml_client = mlflow.MlflowClient(tracking_uri="http://0.0.0.0:5151")
mlflow.set_tracking_uri("http://0.0.0.0:5151")
exp_name = 'evaluation_model'
exp_tags = {'project_name':'evaluation_model',"exp_name":exp_name,"mlflow.note.content": "evaluation of rag for comparison"}
#mlflow.delete_experiment(experiment.experiment_id)
experiment = ml_client.get_experiment_by_name(exp_name)
if experiment is None:
  ml_client.create_experiment(name=exp_name,tags=exp_tags)
  experiment = ml_client.get_experiment_by_name(exp_name)
mlflow.set_experiment(exp_name)

thresholds = {"accuracy_score": MetricThreshold(threshold=0.8,min_absolute_change=0.05,min_relative_chx`ange=0.05,greater_is_better=True)}

signature = infer_signature(ansDf['ref_answer'],ansDf['pred_answer'])

with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
  # mlflow.sklearn.log_model(model, "model", signature=signature)
  # model_uri = mlflow.get_artifact_uri("model")
  resL = mlflow.evaluate(data=ansDf,targets="pred_answer",predictions="ref_answer",model_type="classifier",evaluators="default")



artL, runL = [], []
artifactN = "evaluate_openai"
modelN = "gpt-4o-mini"
llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
  resL = mlflow.evaluate(data=ansDf,targets="pred_answer",predictions="ref_answer"#,model_type="question_answering"
                         ,extra_metrics=[mlflow.metrics.exact_match(),rougeLsum(),rouge2(),mlflow.metrics.toxicity(), mlflow.metrics.latency()]#,answer_similarity(), answer_relevance()]
                         ,evaluators="default")
  runL.append(mlflow.active_run().info.run_id)
  artL.append(artifactN)
  #mlflow.shap.log_explanation(model.predict, X)   
  print(f"See aggregated evaluation results below: \n{resL.metrics}")


with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
  mlflow.evaluate(eval_data,targets="truth",predictions='answer',model_type="classifier",validation_thresholds=thresholds,)


answer_similarity_metric = answer_similarity(model="openai:/gpt-4")


with mlflow.start_run() as run:
  candidate_model_uri = mlflow.sklearn.log_model(candidate_model,"candidate_model",signature=signature).model_uri
  baseline_model_uri = mlflow.sklearn.log_model(baseline_model,"baseline_model",signature=signatur.model_urie)
  mlflow.evaluate(candidate_model_uri,eval_data,targets="label",model_type="classifier",validation_thresholds=thresholds,baseline_model=baseline_model_uri)

  
eval_table = resL.tables["eval_results_table"]
print(f"See evaluation table below: \n{eval_table}")


from mlflow.deployments import set_deployments_target
set_deployments_target("http://localhost:5151")

  


#--------------------------------------ragas-qeval------------------------------

from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall, context_precision
from sacrerouge.metrics import QAEval


eval_df = ansDf[['question','pred_answer','context']]
eval_df.columns = ['question','answer','contexts']
score = evaluate(dataset,metrics=[faithfulness,answer_correctness,context_recall,context_precision])
score.to_pandas()

qaeval = QAEval()

scores, qas = qaeval.score(eval_df['ref_answer'],[eval_df['pred_answer']],return_qa_pairs=True)

import evaluate
precision_metric = evaluate.load("precision")
squad_metric = evaluate.load("squad")
rouge = evaluate.load('rouge')
results = precision_metric.compute(references=ansDf['truth'], predictions=ansDf['answer'])
results = rouge.compute(references=ansDf['ref_answer'], predictions=ansDf['pred_answer'])

#-------------------------------------prompquality----------------------------------

import promptquality as pq
from promptquality import Scorers
all_metrics = [Scorers.latency,Scorers.pii,Scorers.toxicity,Scorers.tone,Scorers.context_adherence,Scorers.completeness_gpt,Scorers.chunk_attribution_utilization_gpt]


from llama_index.evaluation import BatchEvalRunner
from llama_index.evaluation import (FaithfulnessEvaluator,RelevancyEvaluator)
service_context_gpt4 = ...
vector_index = ...
question_list = ...

faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=8,
)eval_results = runner.evaluate_queries(
    vector_index.as_query_engine(), queries=question_list
)

#------------------------------------bedrock-----------------------------------------


import boto3
client = boto3.client('bedrock')

job_request = client.create_evaluation_job(jobName="api-auto-job-titan",
    jobDescription="two different task types",roleArn="arn:aws:iam::111122223333:role/role-name",
    inferenceConfig={"models": [{"bedrockModel": {"modelIdentifier":"arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-text-lite-v1","inferenceParams":"{\"temperature\":\"0.0\", \"topP\":\"1\", \"maxTokenCount\":\"512\"}"}}]},
    outputDataConfig={"s3Uri":"s3://model-evaluations/outputs/"},
    evaluationConfig={"automated": {"datasetMetricConfigs": [{"taskType": "QuestionAndAnswer","dataset": {"name": "Builtin.BoolQ"},"metricNames": ["Builtin.Accuracy","Builtin.Robustness"]}]}})

print(job_request)

#---------------------------------------cross-evaluation-heuristic----------------------------

professionalism_example_score_2 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps you track experiments, package your code and models, and collaborate with your team, making the whole ML workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and exclamation points, which make it sound less professional. "
    ),
)
professionalism_example_score_4 = mlflow.metrics.genai.EvaluationExample(
    input="What is MLflow?",
    output=("MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.",),
    score=4,
    justification=("The response is written in a formal language and a neutral tone. "),
)

professionalism = mlflow.metrics.genai.make_genai_metric(
    name="professionalism",
    definition=("Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is tailored to the context and audience. It often involves avoiding overly casual language, slang, or colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=("Professionalism: If the answer is written using a professional tone, below are the details for different scores: - Score 0: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for professional contexts."
        "- Score 1: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 2: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 3: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 4: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal business or academic settings. "
    ),
    examples=[professionalism_example_score_2, professionalism_example_score_4],
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    greater_is_better=True,
)









import requests
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
import weaviate
from weaviate.embedded import EmbeddedOptions
url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
client = weaviate.Client(embedded_options = EmbeddedOptions())
vectorstore = Weaviate.from_documents(client = client,documents = chunks,embedding = OpenAIEmbeddings(),by_text = False)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = ({"context": retriever,  "question": RunnablePassthrough()} | prompt | llm | StrOutputParser() )

questions = ["What did the president say about Justice Breyer?", 
             "What did the president say about Intel's CEO?",
             "What did the president say about gun violence?",
            ]
ground_truths = [["The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                ["The president asked Congress to pass proven measures to reduce gun violence."]]
answers = []
contexts = []

for query in questions:
  answers.append(rag_chain.invoke(query))
  contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])
data = {"question": questions,"answer": answers,"contexts": contexts,"ground_truths": ground_truths}
dataset = Dataset.from_dict(data)







