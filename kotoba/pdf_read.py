#https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools
#https://datascience.blog.wzb.eu/category/pdfs/
import os, sys, json, re, pathlib
import base64, io
import subprocess
import numpy as np
import pandas as pd
import requests

subprocess.run(["echo","$VIRTUAL_ENV"],shell=True)
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
fName = "foo"
fName = "am35"
fName = "iplex_nx"
fName = "AM5386"
#fName = "Policies"
fPath = baseDir + fName + '.pdf'
fUrl = "https://www.olympus-ims.com/en/rvi-products/iplex-nx/#!cms[focus]=cmsContent13653"

#-------------------------------------------------unstructured-----------------------------------
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader(fPath, mode="elements")
data = loader.load()

from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from PIL import Image

elements = partition_pdf(filename=fPath,extract_images_in_pdf=True,infer_table_structure=True,chunking_strategy="by_title",max_characters=4000,new_after_n_chars=3800,combine_text_under_n_chars=2000,image_output_dir_path=baseDir+"pdfImages/")

llm = ChatOpenAI(model="gpt-4-vision-preview")
def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')

image_str = image_to_base64("static/pdfImages/figure-15-6.jpg")
chat = ChatOpenAI(model="gpt-4-vision-preview",max_tokens=1024)
msg = chat.invoke([HumanMessage(content=[{"type": "text", "text" : "Please give a summary of the image provided. Be descriptive"},{"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{image_str}"},},])])
msg.content

#-------------------------------------pypdfium2-------------------------------------------------
from langchain_community.document_loaders import PyPDFium2Loader
loader = PyPDFium2Loader(fPath)
data = loader.load()

#----------------------------------------pdfminer------------------------------------------------

from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader

loader = PDFMinerPDFasHTMLLoader(fPath)
data = loader.load()

#-----------------------------------------texatract----------------------------------------------

from langchain_community.document_loaders import AmazonTextractPDFLoader
from textractor.data.constants import TextractFeatures
from textractor import TExtractor
from textractor import Textractor


loader = AmazonTextractPDFLoader(baseDir + "szx7.png")
documents = loader.load()
extractor = TExtractor(profile_name="default")
document = extractor.analyze_document(
	file_source=baseDir + "szx7.png",
	features=[TextractFeatures.TABLES]
)
document.tables[0].to_excel(baseDir+"output.xlsx")

extractor = Textractor(profile_name="default")
from textractor.data.constants import TextractFeatures
document = extractor.analyze_document(
    file_source="tests/fixtures/form.png",
    features=[TextractFeatures.TABLES]
)
document.tables[0].to_excel("output.xlsx")


#-----------------------------------------azure------------------------------------------------

%pip install --upgrade --quiet  langchain langchain-community azure-ai-documentintelligence
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
loader = AzureAIDocumentIntelligenceLoader(api_endpoint="", api_key="", file_path=fPath, api_model="prebuilt-layout")
documents = loader.load()

#-------------------------------------------upstage---------------------------------------------

from langchain_upstage import UpstageLayoutAnalysisLoader
os.environ["UPSTAGE_DOCUMENT_AI_API_KEY"] = "YOUR_API_KEY"
loader = UpstageLayoutAnalysisLoader(fPath)
data = loader.load()

#----------------------------------------------agent-chunking-------------------------------------

from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel
from langchain import hub

obj = hub.pull("wfh/proposal-indexing")
llm = ChatOpenAI(model='gpt-4-1106-preview', openai_api_key = os.getenv("OPENAI_API_KEY", 'YouKey'))
runnable = obj | llm

class Sentences(BaseModel):
    sentences: List[str]
    
extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
def get_propositions(text):
    runnable_output = runnable.invoke({"input": text}).content
    propositions = extraction_chain.run(runnable_output)[0].sentences
    return propositions

with open(baseDir + "AM5386" + '.txt') as f:
    essay = f.read()

paragraphs = essay.split("\n\n")
len(paragraphs)
essay_propositions = []
for i, para in enumerate(paragraphs[:5]):
    propositions = get_propositions(para)
    essay_propositions.extend(propositions)
    print (f"Done with {i}")

print (f"You have {len(essay_propositions)} propositions")
essay_propositions[:10]

#------------------------------------mathpix----------------------------------------------------

from langchain_community.document_loaders import MathpixPDFLoader
loader = MathpixPDFLoader(fPath)

#------------------------------------diffbot--------------------------------------------------------

from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
diffbot_nlp = DiffbotGraphTransformer(diffbot_api_key=os.getenv("DIFFBOT_API_KEY", 'YourKey'))
text = """
Greg is friends with Bobby. San Francisco is a great city, but New York is amazing.
Greg lives in New York. 
"""
docs = [Document(page_content=text)]
graph_documents = diffbot_nlp.convert_to_graph_documents(docs)
graph_documents
    
#-------------------------------------------------tika-------------------------------------------

import tika
tika.initVM()
from tika import parser, detector
parsed = parser.from_file(fPath,xmlContent=True)
print(parsed["content"])
print(detector.from_file(fPath))

#---------------------------------------------------pymupdf---------------------------------------

import pymupdf
import pymupdf4llm
import markdown
with pymupdf.open(fPath) as doc:  
    text = chr(12).join([page.get_text() for page in doc])

pathlib.Path(baseDir + fName + ".txt").write_bytes(text.encode())
md_text = pymupdf4llm.to_markdown(fPath)
pathlib.Path(baseDir + fName + ".md").write_bytes(md_text.encode())
html_text = markdown(md_text,extensions=['markdown.extensions.tables'])
pathlib.Path(baseDir + fName + ".html").write_bytes(html_text.encode())

#---------------------------------------beatifulsoup---------------------------------------------

from bs4 import BeautifulSoup
with open(baseDir + fName + '.html') as fByte:
    fString = fByte.read()
response = requests.get(fUrl) 

with open(baseDir + 'iplex.html','w') as fByte:
    fByte.write(response.text)

soup = BeautifulSoup(response.text, 'html.parser')
tableL = soup.find_all('table')
tableS = "".join([str(t) for t in tableL])
tabDf = pd.read_html(tableS)
for tab in tableL:
    t = str(tab)
    if re.search("flexibility gradually",t):
        tabD  = pd.read_html(t, header=[0,1])[0]
        break
tabD.to_csv(baseDir + "implex.csv",index=False)

#------------------------------------------pdftabextract------------------------------------------

from pdftabextract import imgproc
from pdftabextract.common import read_xml, parse_pages
from math import radians, degrees
from pdftabextract.common import ROTATION, SKEW_X, SKEW_Y
from pdftabextract.geom import pt
from pdftabextract.textboxes import rotate_textboxes, deskew_textboxes
from pdftabextract.clustering import find_clusters_1d_break_dist
from pdftabextract.clustering import calc_cluster_centers_1d
from pdftabextract.clustering import zip_clusters_and_values
from pdftabextract.textboxes import border_positions_from_texts, split_texts_by_positions, join_texts
from pdftabextract.common import all_a_in_b, DIRECTION_VERTICAL
from pdftabextract.extract import make_grid_from_positions
from pdftabextract.common import save_page_grids
from pdftabextract.extract import fit_texts_into_grid, datatable_to_dataframe

xPath = baseDir + "output.xml"
xmltree, xmlroot = read_xml(xPath)
p_num = 3
p = pages[p_num]
pages = parse_pages(xmlroot)
imgfilebasename = p['image'][:p['image'].rindex('.')]
imgfile = os.path.join(baseDir, p['image'])
print("page %d: detecting lines in image file '%s'..." % (p_num, imgfile))
iproc_obj = imgproc.ImageProc(imgfile)
page_scaling_x = iproc_obj.img_w / p['width']   # scaling in X-direction
page_scaling_y = iproc_obj.img_h / p['height']  # scaling in Y-direction
lines_hough = iproc_obj.detect_lines(canny_kernel_size=3, canny_low_thresh=50, canny_high_thresh=150,
                                     hough_rho_res=1,
                                     hough_theta_res=np.pi/500,
                                     hough_votes_thresh=round(0.2 * iproc_obj.img_w))
print("> found %d lines" % len(lines_hough))
import cv2
def save_image_w_lines(iproc_obj, imgfilebasename):
    img_lines = iproc_obj.draw_lines(orig_img_as_background=True)
    img_lines_file = os.path.join(baseDir, '%s-lines-orig.png' % imgfilebasename)
    
    print("> saving image with detected lines to '%s'" % img_lines_file)
    cv2.imwrite(img_lines_file, img_lines)

save_image_w_lines(iproc_obj, imgfilebasename)
rot_or_skew_type, rot_or_skew_radians = iproc_obj.find_rotation_or_skew(radians(0.5),
                                                                        radians(1),
                                                                        omit_on_rot_thresh=radians(0.5))

needs_fix = True
if rot_or_skew_type == ROTATION:
    print("> rotating back by %f°" % -degrees(rot_or_skew_radians))
    rotate_textboxes(p, -rot_or_skew_radians, pt(0, 0))
elif rot_or_skew_type in (SKEW_X, SKEW_Y):
    print("> deskewing in direction '%s' by %f°" % (rot_or_skew_type, -degrees(rot_or_skew_radians)))
    deskew_textboxes(p, -rot_or_skew_radians, rot_or_skew_type, pt(0, 0))
else:
    needs_fix = False
    print("> no page rotation / skew found")
if needs_fix:
    lines_hough = iproc_obj.apply_found_rotation_or_skew(rot_or_skew_type, -rot_or_skew_radians)
    save_image_w_lines(iproc_obj, imgfilebasename + '-repaired')

output_files_basename = xPath[:xPath.rindex('.')]
repaired_xmlfile = os.path.join(xPath, output_files_basename + '.repaired.xml')
print("saving repaired XML file to '%s'..." % repaired_xmlfile)
xmltree.write(repaired_xmlfile)

MIN_COL_WIDTH = 60
vertical_clusters = iproc_obj.find_clusters(imgproc.DIRECTION_VERTICAL, find_clusters_1d_break_dist,
                                            remove_empty_cluster_sections_use_texts=p['texts'],
                                            remove_empty_cluster_sections_n_texts_ratio=0.1,
                                            remove_empty_cluster_sections_scaling=page_scaling_x,
                                            dist_thresh=MIN_COL_WIDTH/2)
print("> found %d clusters" % len(vertical_clusters))
img_w_clusters = iproc_obj.draw_line_clusters(imgproc.DIRECTION_VERTICAL, vertical_clusters)
save_img_file = os.path.join(baseDir, '%s-vertical-clusters.png' % imgfilebasename)
print("> saving image with detected vertical clusters to '%s'" % save_img_file)
cv2.imwrite(save_img_file, img_w_clusters)
page_colpos = np.array(calc_cluster_centers_1d(vertical_clusters)) / page_scaling_x
print('found %d column borders:' % len(page_colpos))
print(page_colpos)
col2_rightborder = page_colpos[2]
median_text_height = np.median([t['height'] for t in p['texts']])
text_height_deviation_thresh = median_text_height / 2
texts_cols_1_2 = [t for t in p['texts']
                  if t['right'] <= col2_rightborder
                     and abs(t['height'] - median_text_height) <= text_height_deviation_thresh]
borders_y = border_positions_from_texts(texts_cols_1_2, DIRECTION_VERTICAL)
clusters_y = find_clusters_1d_break_dist(borders_y, dist_thresh=median_text_height/2)
clusters_w_vals = zip_clusters_and_values(clusters_y, borders_y)
pos_y = calc_cluster_centers_1d(clusters_w_vals)
pos_y.append(p['height'])
print('number of line positions:', len(pos_y))
pttrn_table_row_beginning = re.compile(r'^[\d Oo][\d Oo]{2,} +[A-ZÄÖÜ]')
texts_cols_1_2_per_line = split_texts_by_positions(texts_cols_1_2, pos_y, DIRECTION_VERTICAL,
                                                   alignment='middle',
                                                   enrich_with_positions=True)
for line_texts, (line_top, line_bottom) in texts_cols_1_2_per_line:
    line_str = join_texts(line_texts)
    if pttrn_table_row_beginning.match(line_str):  
        top_y = line_top
        break
else:
    top_y = 0

words_in_footer = ('anzeige', 'annahme', 'ala')
min_footer_text_height = median_text_height * 1.5
min_footer_y_pos = p['height'] * 0.7
bottom_texts = [t for t in p['texts']
                if t['top'] >= min_footer_y_pos and t['height'] >= min_footer_text_height]
bottom_texts_per_line = split_texts_by_positions(bottom_texts,
                                                 pos_y + [p['height']],
                                                 DIRECTION_VERTICAL,
                                                 alignment='middle',
                                                 enrich_with_positions=True)
page_span = page_colpos[-1] - page_colpos[0]
min_footer_text_width = page_span * 0.8
for line_texts, (line_top, line_bottom) in bottom_texts_per_line:
    line_str = join_texts(line_texts)
    has_wide_footer_text = any(t['width'] >= min_footer_text_width for t in line_texts)
    if has_wide_footer_text or all_a_in_b(words_in_footer, line_str):
        bottom_y = line_top
        break
else:
    bottom_y = p['height']

page_rowpos = [y for y in pos_y if top_y <= y <= bottom_y]
print("> page %d: %d lines between [%f, %f]" % (p_num, len(page_rowpos), top_y, bottom_y))
grid = make_grid_from_positions(page_colpos, page_rowpos)
n_rows = len(grid)
n_cols = len(grid[0])
print("> page %d: grid with %d rows, %d columns" % (p_num, n_rows, n_cols))
page_grids_file = os.path.join(baseDir, output_files_basename + '.pagegrids_p3_only.json')
print("saving page grids JSON file to '%s'" % page_grids_file)
save_page_grids({p_num: grid}, page_grids_file)
datatable = fit_texts_into_grid(p['texts'], grid)
df = datatable_to_dataframe(datatable)
df.head(n=10)
csv_output_file = os.path.join(baseDir, output_files_basename + '-p3_only.csv')
print("saving extracted data to '%s'" % csv_output_file)
df.to_csv(csv_output_file, index=False)
excel_output_file = os.path.join(baseDir, output_files_basename + '-p3_only.xlsx')
print("saving extracted data to '%s'" % excel_output_file)
df.to_excel(excel_output_file, index=False)


#------------------------------------------table-extract-------------------------------------------
import pdftableextract as pdf
root, ext = os.path.splitext(os.path.basename(fPath))
pages = ['1']
cells = [pdf.process_page(sys.argv[1], p) for p in pages]
cells = [cell for row in cells for cell in row]

tables = pdf.table_to_list(cells, pages)
for i, table in enumerate(tables[1:]):
    df = pd.DataFrame(table)
    out = '{}-page-1-table-{}.csv'.format(root, i + 1)
    df.to_csv(out, index=False, quoting=1, encoding='utf-8')

#-------------------------------pdftables------------------------------------------------
resq = requests.post("https://pdftables.com/api?key="+os.environ['PDFTABLES_KEY']+"&format=xlsx-single")


#-------------------------------tika--------------------------------------------

import tika
tika.initVM()
from tika import parser
parsed = parser.from_file(fPath)
print(parsed["metadata"])
print(parsed["content"])
                     
#----------------------------pypdf------------------------------------------------
from pypdf import PdfReader
reader = PdfReader(fPath)
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()

#----------------------------llmsherpa-------------------------------------------

from llmsherpa.readers import LayoutPDFReader
pdf_reader = LayoutPDFReader("https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all")
doc = pdf_reader.read_pdf(fPath)
docL = []
for s in doc.sections():
    sectS = ''
    for p in s.children:
        sectS += p.to_text()
        if sectS == '':
            sectS = '-'
        docL.append(Document(text=sectS,metadata={"sect":s.to_context_text(),"lev":s.level}))
for t in doc.tables():
    docL.append(Document(text=t.to_text(),metadata={"table":s.block_idx,"lev":t.level}))

#---------------------------------------------pymupdf---------------------------

import pymupdf4llm
import pymupdf
md_text = pymupdf4llm.to_markdown(pdf_doc,pages=[0,1])
md_text = pymupdf4llm.to_markdown(pdf_doc)
# parser = LlamaParse(api_key="...",result_type="markdown")
# documents = parser.load_data("./my_file.pdf") 
#single_sentences_list = re.split(r'(?<=[.?!])\s+', essay)
headers_split = [("#", "Chapter"),("##", "Section"),('###','Subsection')]
splitter = MarkdownHeaderTextSplitter(headers_split)#,strip_headers=True,return_each_line=False,)
docL = splitter.split_text(md_text)
#splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
#splitter = SentenceSplitter(chunk_size=200,chunk_overlap=15)
#elements = partition_pdf(filename=pdf_doc,strategy="hi_res",infer_table_structure=True,model_name="yolox")

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-"
llm = get_llm()
parsing_instructions = '''The document describes IT security policies for audit. It contains many tables. Answer questions using the information in this article and be precise.'''
documents = LlamaParse(result_type="markdown", parsing_instructions=parsing_instructions).load_data(pdf_doc)
print(documents[0].text[:1000])
node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8).from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

#-------------------------------------------pypdf2------------------------------

from PyPDF2 import PdfReader
text = ""
docL = []
for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        docL.append(Document(text=text,metadata={"page":i}))
        


#-----------------------------------camelot-----------------------------

import camelot
tables = camelot.read_pdf(fPath)
tDf = tables[0].df
tDf.to_csv(baseDir + fName + ".csv")

#----------------------------------pdf-plumber-------------------------------

import fitz 
import pdfplumber
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Preformatted

font_size_counter = Counter()
with pdfplumber.open(fPath) as pdf:
    for i in range(len(pdf.pages)):
        words = pdf.pages[i].extract_words(extra_attrs=['fontname', 'size'])
        lines = {}
        for word in words:
            line_num = word['top']
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(word)
        for line_words in lines.values():
            font_size_counter[line_words[0]['size']] += 1

repeated_sizes = [size for size, count in font_size_counter.items() if count > 1]
extracted_font_size = max(repeated_sizes)

chunks = extract_chunks_from_pdf(fPath, markers)


lines_with_target_font_size = []
with pdfplumber.open(fPath) as pdf:
    for i in range(len(pdf.pages)):
        words = pdf.pages[i].extract_words(extra_attrs=['fontname', 'size'])
        lines = {}
        for word in words:
            line_num = word['top']
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(word)
        for line_num, line_words in lines.items():
            line_font_sizes = [word['size'] for word in line_words]
            if target_font_size in line_font_sizes:
                line_text = ' '.join([word['text'] for word in line_words])
                lines_with_target_font_size.append(line_text)

extracted_font_size = lines_with_target_font_size
                
doc = SimpleDocTemplate(output_fPath, pagesize=letter)
styles = getSampleStyleSheet()
story = []
for chunk in chunks:
    preformatted = Preformatted(chunk, styles["Normal"])
    story.append(preformatted)
doc.build(story)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i, chunk in enumerate(chunks, start=1):
    output_fPath = os.path.join(output_folder, f"output_pdf_part{i}.pdf")
    write_chunks_to_pdf([chunk], output_fPath)

chunks = []
current_chunk = []
current_marker_index = 0
pdf_document = fitz.open(fPath)
for page_num in range(pdf_document.page_count):
    page = pdf_document[page_num]
    text = page.get_text("text")
    lines = text.split('\n')
    for line in lines:
        if current_marker_index < len(markers) and markers[current_marker_index] in line:
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_marker_index += 1
        current_chunk.append(line)
if current_chunk:
    chunks.append('\n'.join(current_chunk))
pdf_document.close()
output_folder = "output"

print("te se qe te ve be te ne?")

#https://www.jnjmedtech.com/system/files/pdf/090912-220322%20DSUS_EMEA%20Large%20Bone%20Saw%20Blades%20Product%20Brochure.pdf
