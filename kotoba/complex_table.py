import os, sys, json, re
import pandas as pd
import streamlit
import langchain as lc
from tabula import read_pdf
import camelot
import PyPDF2
import pandasai
from pandasai import SmartDataframe
from textractor import Textractor
# import pdftotree $ with tensorflow

os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.chatbot_utils as c_t
import importlib

baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
modL = ["gpt-4o@openai","gpt-4-turbo@openai","gpt-3.5-turbo@openai","mixtral-8x7b-instruct-v0.1@aws-bedrock","llama-2-70b-chat@aws-bedrock","codellama-34b-instruct@together-ai","gemma-7b-it@fireworks-ai","claude-3-haiku@anthropic","claude-3-opus@anthropic","claude-3-sonnet@anthropic","mistral-7b-instruct-v0.1@fireworks-ai","mistral-7b-instruct-v0.2@fireworks-ai"]

llm = pandasai.llm.openai.OpenAI(api_token=os.environ['OPENAI_API_KEY'])
tabL = camelot.read_pdf(baseDir + 'foo.pdf')
fooD = read_pdf(baseDir + 'china.pdf')
# page = pdftotree.parse(baseDir + 'china.pdf',html_path=None,model_type=None,model_path=None,visualize=False)
tab = tabL[0].df
agent = pandasai.Agent(tab, config={"llm": llm})
fooD = SmartDataframe(tab, config={"llm": llm})

fooD.chat("what is the average distance?")
agent.chat('Which are the top 5 countries by sales?')

importlib.reload(c_t)
unify = c_t.get_unify(modL[0])

if False: # table from image
  extractor = Textractor(profile_name="default")
  from textractor.data.constants import TextractFeatures
  document = extractor.analyze_document(
    file_source="tests/fixtures/form.png",
    features=[TextractFeatures.TABLES]
  )
  document.tables[0].to_excel("output.xlsx")

