import re, json, os, sys
import argparse
import logging
import instructor
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from graphviz import Digraph
import kotoba.knowledge_structure as k_s

instructor.patch()
load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
response_data = ""
driver = Neo4j()
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
fName = "foo"
fName = "am35"
fName = "iplex_nx"
fName = "AM5386"
#fName = "Policies"
fPath = baseDir + fName + '.pdf'
fUrl = "https://www.olympus-ims.com/en/rvi-products/iplex-nx/#!cms[focus]=cmsContent13653"

with open(baseDir + fName + '.html') as fByte:
    fString = fByte.read()
response = requests.get(fUrl) 
soup = BeautifulSoup(response.text, "html.parser")
paragraphs = soup.find_all("p")
text = " ".join([p.get_text() for p in paragraphs])

user_input = "spark"
openai.api_key = os.environ['OPENAI_API_KEY']
prompt = f"Help me understand following by describing as a detailed knowledge graph: {user_input}"
completion: KnowledgeGraph = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",messages=[{"role": "user","content": prompt,}],response_model=KnowledgeGraph,)
response_data = completion.model_dump()
edges = response_data["edges"]
def _restore(e):
    e["from"] = e["from_"]
    return e

response_data["edges"] = [_restore(e) for e in edges]
results = driver.get_response_data(response_data)

dot = Digraph(comment="Knowledge Graph")
response_dict = response_data
for node in response_dict.get("nodes", []):
    dot.node(node["id"], f"{node['label']} ({node['type']})")

for edge in response_dict.get("edges", []):
    dot.edge(edge["from"], edge["to"], label=edge["relationship"])

dot.render("knowledge_graph.gv", view=False)
dot.format = "png"
dot.render("static/knowledge_graph", view=False)
png_url = f"{request.url_root}static/knowledge_graph.png"

(nodes, edges) = driver.get_graph_data()
response_dict = response_data
nodes = [
    {
        "data": {
            "id": node["id"],
            "label": node["label"],
            "color": node.get("color", "defaultColor"),
        }
    }
    for node in response_dict["nodes"]
]
edges = [
    {
        "data": {
            "source": edge["from"],
            "target": edge["to"],
            "label": edge["relationship"],
            "color": edge.get("color", "defaultColor"),
        }
    }
    for edge in response_dict["edges"]
]
graphD = jsonify({"elements": {"nodes": nodes, "edges": edges}})


