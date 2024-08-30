import os, sys, json, re
import pandas as pd
import langchain as lc
import camelot
import pandasai
# import pdftotree $ with tensorflow

os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.chatbot_utils as c_t
import importlib
import markdown
from bs4 import BeautifulSoup


baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
fName = "AM5386"
modL = ["gpt-4o@openai","gpt-4-turbo@openai","gpt-3.5-turbo@openai","mixtral-8x7b-instruct-v0.1@aws-bedrock","llama-2-70b-chat@aws-bedrock","codellama-34b-instruct@together-ai","gemma-7b-it@fireworks-ai","claude-3-haiku@anthropic","claude-3-opus@anthropic","claude-3-sonnet@anthropic","mistral-7b-instruct-v0.1@fireworks-ai","mistral-7b-instruct-v0.2@fireworks-ai"]
os.environ['OPENAI_MODEL_NAME'] = modL[0]

llm = pandasai.llm.openai.OpenAI(api_token=os.environ['UNIFY_KEY'],model=modL[0],base_url="https://api.unify.ai/v0/")
with open(baseDir + fName + ".html") as fByte:
    html_text = fByte.read()
soup = BeautifulSoup(html_text, 'html.parser')
tableL = soup.find_all('table')
tableS = "".join([str(t) for t in tableL])
tabDf = pd.read_html(tableS)
for tab in tableL:
    t = str(tab)
    if re.search("flexibility gradually",t):
        tabD  = pd.read_html(t, header=[0,1])[0]
        break

agent = pandasai.Agent(tabD, config={"llm": llm})
fooD = pandasai.SmartDataframe(tabD, config={"llm": llm})

fooD.chat("what is the average distance?")
agent.chat('Which are the top 5 countries by sales?')


sales_by_country = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "sales": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})
agent = pandasai.Agent(sales_by_country, config={"llm": llm})
agent.chat('Which are the top 5 countries by sales?')


from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create (
   model="gpt-3.5-turbo",
   messages=[
       {"role": "user", "content": "Who was the first man on the moon?"},
       {"role": "assistant", "content": "The first man on the moon was Neil Armstrong."},
       {"role": "user", "content": "Tell me more about him."}
   ],
   top_p=0.5
 )
