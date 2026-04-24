from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import nomic
import fitz  # PyMuPDF
from nomic import embed
from pinecone import Pinecone, Vector
import os
from dotenv import load_dotenv
import uuid
import requests
from bs4 import BeautifulSoup
import base64
import io

from PIL import Image
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from groq import Groq


base_dir = os.path.dirname(os.path.abspath(__file__))
cookie_path = os.path.join(base_dir, 'youtube_cookies.txt')


load_dotenv()

# -------- CLIENTS --------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

import google.generativeai as genai # 

# Client initialize karein
client_gemini = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")

if NOMIC_API_KEY:
    nomic.login(NOMIC_API_KEY)
else:
    print("Warning: NOMIC_API_KEY not found!")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("context-hub")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- NAMESPACES --------
PDF_NAMESPACE = "pdf"  
current_pdf_namespace = "pdf_default"
IMAGE_NAMESPACE = "image"
YOUTUBE_NAMESPACE = "youtube"

# -------- HELPER FUNCTIONS --------

def chunk_text(text: str, size: int = 500):
    return [text[i:i+size] for i in range(0, len(text), size)]


def create_embeddings(chunks: list):
    if not chunks:
        return []
    try:
        response = embed.text(
            texts=chunks,
            model="nomic-embed-text-v1",
            task_type="search_document"
        )
        result = dict(response)
        if 'embeddings' in result and len(result['embeddings']) > 0:
            return result['embeddings']
        else:
            print("DEBUG: Nomic returned no embeddings!")
            return []
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


def store_embeddings(chunks: list, embeddings: list, namespace: str = ""):
    if not chunks or not embeddings:
        print("DEBUG: Nothing to store.")
        return
    vectors = [
        Vector(
            id=str(uuid.uuid4()),
            values=embeddings[i],
            metadata={"text": chunks[i]}
        )
        for i in range(len(embeddings))
    ]
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)   # ← namespace added


def search(query: str, namespace: str = ""):
    try:
        query_response = embed.text(
            texts=[query],
            model="nomic-embed-text-v1",
            task_type="search_query"
        )
        result = dict(query_response)
        query_emb = result['embeddings'][0]

        results = index.query(
            vector=query_emb,
            top_k=3,
            include_metadata=True,
            namespace=namespace    # ← namespace added
        )
        relevant_chunks = []
        for match in results['matches']:
            if match.get('metadata') and 'text' in match['metadata']:
                relevant_chunks.append(match['metadata']['text'])
        return relevant_chunks

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_answer(query: str, context: str):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer ONLY using the provided context. "
                        "If the answer is not in the context, say you don't know. "
                        "When you find an answer, elaborate and provide more information."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Groq Error: {str(e)}")
        return f"Groq Error: {str(e)}"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_namespace
    try:
        # Generate a fresh unique namespace for every new upload
        current_pdf_namespace = f"pdf_{uuid.uuid4().hex}"
        print(f"DEBUG: Using new namespace: {current_pdf_namespace}")

        content = await file.read()
        pdf = fitz.open(stream=content, filetype="pdf")

        text = ""
        for page in pdf:
            text += page.get_text()

        if not text.strip():
            return {"error": "PDF is empty or could not be read."}

        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        store_embeddings(chunks, embeddings, namespace=current_pdf_namespace)
        return {"message": "PDF processed successfully."}

    except Exception as e:
        print(f"Upload Error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}
    
    
@app.post("/query")
async def query_pdf(query: str = Query(...)):
    global current_pdf_namespace
    try:
        results = search(query, namespace=current_pdf_namespace)
        if not results:
            return {"answer": "I couldn't find any relevant information in the document."}
        context = " ".join(results)
        answer = generate_answer(query, context)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
    
# -------- IMAGE HELPERS --------

def create_image_embedding(image):
    output = embed.image(
        images=[image],
        model="nomic-embed-vision-v1.5"
    )
    result = dict(output)
    return result["embeddings"][0]


def store_image_embedding(image_vector, filename: str):
    index.upsert(
        vectors=[Vector(
            id=str(uuid.uuid4()),
            values=image_vector,
            metadata={"type": "image", "filename": filename}
        )],
        namespace=IMAGE_NAMESPACE
    )

def generate_image_answer(query, image):
    try:
        # Convert image → bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Convert to base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        response = client_gemini.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": query},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        )

        return response.text if response.text else "No answer generated."

    except Exception as e:
        return f"Error: {str(e)}"
    


# -------- URL HELPERS --------

def extract_text_from_url(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return text[:5000]
    except Exception as e:
        print(f"URL Error: {e}")
        return None


def generate_url_answer(url: str, query: str = None):
    text = extract_text_from_url(url)
    if not text:
        return "Could not fetch content from this URL."
    try:
        user_msg = (
            f"URL: {url}\n\n"
            f"Content:\n{text}\n\n"
            f"Task: Explain what this page is about in simple words."
        )
        if query:
            user_msg += f"\n\nUser Question: {query}"

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about webpage content."
                },
                {
                    "role": "user",
                    "content": user_msg
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


# -------- ROUTES --------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        try:
            index.delete(delete_all=True, namespace="")
            print("DEBUG: Pinecone default namespace cleared.")
        except Exception as e:
            print(f"DEBUG: Delete skipped: {e}")

        content = await file.read()
        pdf = fitz.open(stream=content, filetype="pdf")

        text = ""
        for page in pdf:
            text += page.get_text()

        if not text.strip():
            return {"error": "PDF is empty or could not be read."}

        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        store_embeddings(chunks, embeddings)
        return {"message": "PDF processed successfully."}

    except Exception as e:
        print(f"Upload Error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}


@app.post("/query")
async def query_pdf(query: str = Query(...)):
    try:
        results = search(query)
        if not results:
            return {"answer": "I couldn't find any relevant information in the document."}
        context = " ".join(results)
        answer = generate_answer(query, context)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


current_image = None

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    global current_image
    try:
        image = Image.open(file.file)
        current_image = image
        image_vector = create_image_embedding(image)
        store_image_embedding(image_vector, file.filename)
        return {"message": "Image processed successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/query-image")
async def query_image(query: str = Query(...)):
    global current_image
    try:
        if current_image is None:
            return {"error": "No image uploaded"}
        answer = generate_image_answer(query, current_image)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


current_url = None
current_url_text = None

@app.post("/process-url")
async def process_url(url: str = Query(...)):
    global current_url, current_url_text
    text = extract_text_from_url(url)
    if not text:
        return {"error": "Failed to extract content"}
    current_url = url
    current_url_text = text
    return {"message": "URL processed successfully"}


@app.post("/query-url")
async def query_url(query: str = Query(...)):
    global current_url, current_url_text
    if not current_url_text:
        return {"error": "No URL processed"}
    try:
        answer = generate_url_answer(current_url, query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
# ####################################################################################



# -------- YOUTUBE HELPERS --------

def get_video_id(url: str):
    parsed_url = urlparse(url)
    if "youtube.com" in url:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif "youtu.be" in url:
        return parsed_url.path.strip("/")
    return None


# from youtube_transcript_api import (
#     YouTubeTranscriptApi,
#     TranscriptsDisabled,
#     NoTranscriptFound
# )

# def get_transcript(url: str):
#     try:
#         video_id = get_video_id(url)
#         if not video_id:
#             return "ERROR: Invalid YouTube URL"

#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

#         try:
#             # ✅ Try manual English first
#             transcript = transcript_list.find_transcript(['en'])
#         except:
#             try:
#                 # 🔁 Try auto-generated English
#                 transcript = transcript_list.find_generated_transcript(['en'])
#             except:
#                 # 🔥 FINAL fallback: pick ANY available transcript
#                 for t in transcript_list:
#                     try:
#                         transcript = t
#                         break
#                     except:
#                         continue
#                 else:
#                     return "ERROR: No transcripts available"

#         # ✅ Fetch safely
#         try:
#             transcript_data = transcript.fetch()
#         except Exception as e:
#             return f"ERROR: Fetch failed: {str(e)}"

#         text = " ".join([item["text"] for item in transcript_data])

#     except TranscriptsDisabled:
#         return "ERROR: Transcripts are disabled for this video"

#     except Exception as e:
#         return f"ERROR: {str(e)}"


import yt_dlp
import whisper
import random

def get_transcript_whisper(url):
    try:
        # 1. Define the fix specifically for cloud environments like Render
        proxies = [
            "socks5://yxbfpfws:l620a1u85qmi@185.171.254.93:6125",
            "socks5://yxbfpfws:l620a1u85qmi@38.170.172.128:5129",
            "socks5 37.44.219.101 6066 yxbfpfws l620a1u85qmi",
            "socks5 136.0.184.139 6560 yxbfpfws l620a1u85qmi",
            "socks5 82.23.222.52 6358 yxbfpfws l620a1u85qmi",
            "socks5 46.203.206.138 5583 yxbfpfws l620a1u85qmi",
            "socks5 46.203.157.246 7189 yxbfpfws l620a1u85qmi",
            "socks5 91.123.10.157 6699 yxbfpfws l620a1u85qmi",
        ]
        
        # Pick one proxy (helps if one gets rate-limited)
        selected_proxy = random.choice(proxies)

        ydl_opts = {
            'proxy': selected_proxy,
            'socket_timeout': 60,
            'cookiefile': 'cookies.txt',
            'format': 'bestaudio/best',
            'outtmpl': 'audio.%(ext)s',
            'quiet': False,
            'extractor_args': {
                'youtube': {
                    'player_client': ['tv', 'web_creator'],
                }
            },
        }

        print(f"LOG: Attempting download using proxy: {selected_proxy}")


        # Check if the file exists where the code expects it
        if os.path.exists('cookies.txt'):
            print("LOG: cookies.txt found at:", os.path.abspath('cookies.txt'))
            # Optional: Print first line to verify format (Don't print the whole file for security!)
            with open('cookies.txt', 'r') as f:
                print("LOG: Cookie file starts with:", f.readline().strip())
        else:
            print("LOG: ERROR - cookies.txt NOT FOUND!")

        # 2. Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        
        # Check what was actually downloaded
        downloaded_files = glob.glob("audio.*")
        if downloaded_files:
            audio_file = downloaded_files[0]
            print(f"LOG: Successfully downloaded {audio_file}")
            result = model.transcribe(audio_file)
        else:
            return "ERROR: No audio file found after download."

        # 3. Load Whisper model 
        # (Note: Loading this every time is slow; ideally move this outside the function)
        model = whisper.load_model("base") 

        # 4. Transcribe (yt-dlp might download as .m4a or .webm, so we use 'audio.*')
        # To be safe, we'll find the actual file downloaded
        import glob
        audio_files = glob.glob("audio.*")
        if not audio_files:
            return "ERROR: Audio file not found after download."
        
        result = model.transcribe(audio_files[0])

        # Cleanup: Remove the audio file after transcription so you don't fill up Render's disk
        os.remove(audio_files[0])

        return result["text"]

    except Exception as e:
        return f"ERROR: {str(e)}"
    

from pydantic import BaseModel

class YouTubeRequest(BaseModel):
    url: str

@app.post("/process-youtube")
async def process_youtube(request: YouTubeRequest):
    try:
        url = request.url
        video_id = get_video_id(url)

        text = get_transcript_whisper(url)
        if text.startswith("ERROR"):
            print("TRANSCRIPT ERROR:", text)  
            return {"status": "error", "error": text}

        chunks = chunk_text(text, size=1000)
        embeddings = create_embeddings(chunks)

        vectors = [
            Vector(
                id=str(uuid.uuid4()),
                values=embeddings[i],
                metadata={"text": chunks[i]}
            )
            for i in range(len(embeddings))
        ]

        index.upsert(vectors=vectors, namespace=video_id)

        return {"status": "success"}

    except Exception as e:
        return {"error": str(e)}
    


class QueryRequest(BaseModel):
    query: str
    video_id: str

@app.post("/query-youtube")
async def query_youtube(request: QueryRequest):
    try:
        query = request.query
        video_id = request.video_id

        query_response = embed.text(
            texts=[query],
            model="nomic-embed-text-v1",
            task_type="search_query"
        )

        query_emb = dict(query_response)["embeddings"][0]

        results = index.query(
            vector=query_emb,
            top_k=5,
            include_metadata=True,
            namespace=video_id
        )

        context = "\n\n".join([
            match["metadata"]["text"]
            for match in results["matches"]
        ])

        if not context.strip():
            return {"answer": "No relevant info found."}

        answer = generate_answer(query, context)
        return {"answer": answer}
    
    except Exception as e:
        return {"error": str(e)}