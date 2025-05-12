import os
import ast
import re
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyvis.network import Network
from dotenv import load_dotenv


load_dotenv() 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY 
)

system_prompt = SystemMessage(content=(
    "You are a knowledge graph builder. Extract the knowledge from the input text and return only a valid Python list of triples. "
    "Each triple should be in the form (subject, predicate, object). Return the list like [('S', 'P', 'O'), ...]"
))



def load_text(file):
    suffix = ".pdf" if file.name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file.read())
        path = tmp_file.name
    loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return " ".join([chunk.page_content for chunk in chunks])


def extract_triples(text):
    try:
        return ast.literal_eval(text.strip())
    except:
        pattern = re.compile(r"\('.*?'\s*,\s*'.*?'\s*,\s*'.*?'\)")
        matches = pattern.findall(text)
        if matches:
            return [ast.literal_eval(t) for t in matches]
        return []


def display_graph(triples):
    net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='black', directed=True)

    nodes = set()
    for subj, pred, obj in triples:
        if subj not in nodes:
            net.add_node(subj, label=subj, title=subj)
            nodes.add(subj)
        if obj not in nodes:
            net.add_node(obj, label=obj, title=obj)
            nodes.add(obj)
        net.add_edge(subj, obj, label=pred, title=pred)

    net.set_options("""
    var options = {
      "nodes": {
        "font": { "size": 18 },
        "scaling": { "min": 10, "max": 30 },
        "shape": "dot"
      },
      "edges": {
        "arrows": { "to": { "enabled": true }},
        "font": { "size": 14, "align": "middle" },
        "smooth": { "type": "cubicBezier" }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 1
        }
      }
    }
    """)
    html = net.generate_html()
    components.html(html, height=800, scrolling=True)

# Streamlit UI 
st.set_page_config(page_title="Knowledge Graphs", layout="wide")

st.title("Knowledge Graph Generator")
st.markdown("This app extracts knowledge triples from a PDF, text file, or custom text, and visualizes them as an interactive graph.")

st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Choose Input Type:", ["Enter Text", "Upload File"])

if input_mode == "Enter Text":
    user_text = st.text_area("Enter your text below:", height=200)
    input_ready = bool(user_text.strip())
else:
    uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    input_ready = uploaded_file is not None

st.markdown("---")

if st.button(" Generate Knowledge Graph") and input_ready:
    with st.spinner("Extracting knowledge and building graph..."):
        input_text = user_text if input_mode == "Enter Text" else load_text(uploaded_file)
        messages = [system_prompt, HumanMessage(content=input_text)]
        response = model.invoke(messages)
        triples = extract_triples(response.content)

        if triples:
            st.success(f"Extracted {len(triples)} triples.")
            with st.expander("View Extracted Triples"):
                for t in triples:
                    st.write(f"• {t}")
            st.subheader("Knowledge Graph")
            display_graph(triples)
        else:
            st.warning(" No valid knowledge triples found.")
else:
    st.info("Enter input on the left and click the button to generate the graph.")
