#pip install fastapi uvicorn pyngrok nltk requests torch sentence-transformers google-cloud-aiplatform

#pip install "google-cloud-aiplatform[prediction]>=1.16.0"

#pip install nest_asyncio

import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient
from sentence_transformers import SentenceTransformer, util
import torch
import xml.etree.ElementTree as ET
from pyngrok import ngrok
import nest_asyncio
import uvicorn

# Apply the nest_asyncio patch
nest_asyncio.apply()

# Set up environment variables
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/drive/MyDrive/GENASSIS/GENASSIS_credentials.json"
pubmed_api_key = '32141e41b42d3e976957b61c18f6653aa208'
project_id = 'flowing-sign-407814'
region = 'us-central1'
endpoint_id = '7565935223796400128'
aiplatform.init(project=project_id, location=region)

# Initialize FastAPI
app = FastAPI(
    title="GENASSIS PubMed Module",
    description="An API to process research queries and fetch relevant PubMed articles.",
    version="1.0.0",
    contact={
        "name": "Ali Ghoneimy",
        #"url": "https://yourwebsite.com",
        "email": "amghoneimy@gmail.com",
    }
)

# Model for request body
class QueryModel(BaseModel):
    query: str #= Field(..., example="What are the latest advancements in gene therapy?")

def simplify_query(complex_query):
    client = PredictionServiceClient(client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"})

    messages = [
        {"role": "system", "content": f"Modify this query for PubMed Literature search engine. Use synonyms and OR functions to expand search results. ONLY provide the simplified query: {complex_query}"}
    ]

    instances = [{"content": message['content']} for message in messages]
    parameters = {}

    endpoint = client.endpoint_path(project=project_id, location=region, endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)

    simplified_query = response.predictions[0]['content'].strip()
    return simplified_query

def fetch_pubmed_articles(query, start_index=0, max_results=20):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    search_params = {
        'db': 'pubmed',
        'term': query,
        'retstart': start_index,
        'retmax': max_results,
        'api_key': pubmed_api_key,
        'retmode': 'json',
        'sort': 'relevance'
    }

    search_response = requests.get(search_url, params=search_params)
    article_ids = search_response.json().get('esearchresult', {}).get('idlist', [])

    articles = []
    if article_ids:
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(article_ids),
            'api_key': pubmed_api_key,
            'retmode': 'xml',
            'rettype': 'abstract'
        }

        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_response.raise_for_status()

        root = ET.fromstring(fetch_response.content)

        for article in root.findall('.//PubmedArticle'):
            title = article.find('.//ArticleTitle').text
            abstract = article.find('.//AbstractText')
            abstract_text = abstract.text if abstract is not None else 'No abstract available'
            pub_date = article.find('.//PubDate')
            pub_date_text = ET.tostring(pub_date, encoding='unicode', method='text').strip() if pub_date is not None else 'No date available'

            doi = None
            article_ids = article.findall(".//ArticleId")
            for article_id in article_ids:
                if article_id.attrib.get("IdType") == "doi":
                    doi = article_id.text
                    break

            articles.append({'title': title, 'abstract': abstract_text, 'pub_date': pub_date_text, 'doi': doi})

    return articles

def get_most_relevant(query_vector, article_vectors, articles, top_k=20):
    if not articles or len(articles) == 0:
        return []

    similarities = util.cos_sim(query_vector, article_vectors)[0]
    top_k = min(top_k, len(articles))
    top_k_indices = torch.topk(similarities, k=top_k).indices

    most_relevant_articles = [articles[i] for i in top_k_indices]

    return most_relevant_articles

def rag_pipeline(query, relevant_articles):
    context = "\n\n".join([f"Title: {a['title']}\nAbstract: {a['abstract']}\nDOI: {a['doi']}\nDate: {a['pub_date']}" for a in relevant_articles])

    client = PredictionServiceClient(client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"})

    instances = [{"content": f"Context:\n{context}\n\nQuestion: {query}\n\nWith references to the provided context, provide a concise summary in bullet points of the most recent relevant advancements. Include quantitative and qualitative technical details and findings of provided papers."}]
    parameters = {}

    endpoint = client.endpoint_path(project=project_id, location=region, endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)

    return response.predictions[0]['content'].strip()

@app.post("/research", tags=["Research"], summary="Process a research query", description="Processes a research query and fetches relevant PubMed articles.")
def research(query: QueryModel):
    try:
        simplified_query = simplify_query(query.query)
        pubmed_data = fetch_pubmed_articles(simplified_query)

        if not pubmed_data:
            raise HTTPException(status_code=404, detail="No relevant articles found. Please try a different query.")

        texts = [a['title'] + " " + (a.get('abstract', '') or '') for a in pubmed_data]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        article_vectors = model.encode(texts)
        query_vector = model.encode([simplified_query])

        relevant_articles = get_most_relevant(query_vector, article_vectors, pubmed_data)
        answer = rag_pipeline(query.query, relevant_articles)

        return {
            "answer": answer,
            "relevant_articles": relevant_articles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Expose the server using pyngrok
ngrok.set_auth_token("2jw4tDpGE8Ke8uaeglYXhtmQPfk_2n357gKGjBBpwZB61dy9V")
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Run the Uvicorn server
uvicorn.run(app, host="0.0.0.0", port=8000)
