import streamlit as st
import pandas as pd
import json
import requests
import fitz
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from fpdf import FPDF
from neo4j import GraphDatabase
from typing import Dict, Any

# === Load environment variables ===
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1-distill-llama-70b:free"
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
LOG_FILE = "chat_log.json"

# === Streamlit Setup ===
st.set_page_config(page_title="AI CFO Chat", layout="wide")
st.title("AI CFO Chat Interface")

# === PDF Export Utility ===
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "AI CFO Financial Analysis", ln=True, align="C")
    def chapter_body(self, text):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, text)

def export_to_pdf(content: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    path = "ai_cfo_analysis.pdf"
    pdf.output(path)
    return path

# === Neo4j Functions ===
def get_driver():
    try:
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except:
        return None

def get_schema(driver):
    try:
        with driver.session() as session:
            result = session.run("CALL db.schema.visualization()")
            record = result.single()
            return json.dumps({
                "nodes": [dict(labels=labels, properties=list(n.keys())) for n in record["nodes"] for labels in [n.labels]],
                "relationships": [dict(type=r.type, properties=list(r.keys())) for r in record["relationships"]]
            }, indent=2)
    except:
        return ""

def run_cypher(driver, query):
    try:
        with driver.session() as session:
            return [record.data() for record in session.run(query)]
    except:
        return []

# === File Upload & Parsing ===
st.sidebar.header(" Upload Financial Files")
uploaded_files = st.sidebar.file_uploader("Upload CSV, Excel, PDF, JSON, TXT", type=["csv", "xlsx", "xls", "txt", "json", "pdf"], accept_multiple_files=True)

parsed_data: Dict[str, Any] = {}
unreadable_files = []

for file in uploaded_files:
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            parsed_data[file.name] = df.to_dict(orient="records")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
            parsed_data[file.name] = df.to_dict(orient="records")
            if "Department" in df.columns and "Annual Salary" in df.columns:
                st.subheader("Salary by Department")
                plt.figure()
                df.groupby("Department")["Annual Salary"].sum().plot(kind="bar", title="Annual Salary by Department")
                plt.ylabel("Total Salary")
                plt.tight_layout()
                st.pyplot(plt)
        elif name.endswith(".txt"):
            parsed_data[file.name] = file.read().decode("utf-8")[:3000]
        elif name.endswith(".json"):
            try:
                parsed_data[file.name] = json.load(file)
            except json.JSONDecodeError:
                parsed_data[file.name] = file.read().decode("utf-8")[:3000]
        elif name.endswith(".pdf"):
            pdf = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in pdf])
            parsed_data[file.name] = text[:3000]
        else:
            unreadable_files.append(file.name)
    except Exception as e:
        unreadable_files.append(f"{file.name} (Error: {str(e)})")

# === Chat Interface ===
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me about your financials... 🧾")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    driver = get_driver()
    schema_info = get_schema(driver) if driver else ""
    cypher_query = ""
    results = []

    cypher_prompt = f"""
Neo4j Schema:
{schema_info}

User Question:
{user_input}

Generate only a Cypher query.
Do not include code blocks, explanations, or formatting.
"""

    cypher_payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You generate Cypher queries for Neo4j financial graphs."},
            {"role": "user", "content": cypher_prompt}
        ]
    }

    try:
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=cypher_payload)
        cypher_query = response.json()["choices"][0]["message"]["content"].strip()
    except:
        cypher_query = ""

    if cypher_query:
        results = run_cypher(driver, cypher_query)
        if driver:
            driver.close()

    ai_prompt = f"""
User Question:
{user_input}

Neo4j Query Results:
{json.dumps(results, indent=2)}

Uploaded Files:
{json.dumps(parsed_data, indent=2)}

As a virtual CFO, analyze this information and provide insights:
- Summarize financial implications
- Highlight overspending, risks, or inefficiencies
- Provide actionable business recommendations
Avoid technical jargon.
"""

    explain_payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a CFO assistant that explains financial data simply."},
            {"role": "user", "content": ai_prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 2048
    }

    try:
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=explain_payload)
        analysis = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        analysis = f"Error generating analysis: {e}"

    st.session_state.messages.append({"role": "assistant", "content": analysis})
    with st.chat_message("assistant"):
        st.markdown(analysis)

    try:
        chat_log = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                chat_log = json.load(f)
        chat_log.append({"question": user_input, "answer": analysis})
        with open(LOG_FILE, "w") as f:
            json.dump(chat_log, f, indent=2)
    except Exception as e:
        st.warning(f" Log error: {e}")

    with st.expander(" Export this analysis"):
        pdf_path = export_to_pdf(analysis)
        st.download_button(" Download PDF", open(pdf_path, "rb"), file_name="ai_cfo_analysis.pdf")
