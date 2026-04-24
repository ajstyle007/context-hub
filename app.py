import streamlit as st
import requests
import base64
import os
from urllib.parse import urlparse, parse_qs

# --- BACKEND URL ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8002")

# --- PAGE CONFIG ---
st.set_page_config(page_title="ContextHub", layout="wide")

st.markdown("""
    <style>
    .main-header {
        background-color: #427d44;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton>button {
        background-color: #016401;
        color: white;
        width: 100%;
    }
    </style>
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.5rem; 
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            padding-top: 10px;
            padding-bottom: 10px;
            padding-right: 20px;
            padding-left: 20px;
        }
    </style>
    <div class="main-header">
        <h1> 🔍 Context Hub</h1>
    </div>
    """, unsafe_allow_html=True)


tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "About",
    "📄 Document", 
    "🖼️ Visual", 
    "🔗 Web Link", 
    "🎥 Video"
])

with tab0:
    st.markdown("""
            <h3 style='text-align: center; color: #427d44;'>Your Universal Multimodal Intelligence Hub</h3>
            <br>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ⚡ Fast")
        st.write("Powered by **Groq & Gemini** inference for near-instant responses.")
        
    with col2:
        st.markdown("### 🧠 Smart")
        st.write("Uses **Nomic Embeddings** and **Pinecone** for deep context retrieval.")
        
    with col3:
        st.markdown("### 🔄 Versatile")
        st.write("Handles Documents, Images, Web Links, and Videos seamlessly.")

    st.markdown("---")

    st.markdown("## 📖 What is ContextHub?")
    st.write("""
        ContextHub is a state-of-the-art **RAG (Retrieval-Augmented Generation)** platform. 
        Instead of just talking to a general AI, you can give the AI your own specific data—be it a complex 
        PDF report, a website link, or even a video—and ask questions based on that specific content.
    """)

    with st.expander("🚀 How it works under the hood"):
        st.info("""
        1. **Ingest:** You upload a source (PDF, Image, Link, or Video).
        2. **Process:** The system chunks the data and uses **Nomic** to create vector embeddings.
        3. **Store:** These vectors are indexed in a **Pinecone** database.
        4. **Retrieve:** When you ask a question, we find the most relevant context.
        5. **Generate:** **Groq & Gemini** generates an accurate answer based ONLY on your data.
        """)

    st.markdown("---")
    st.markdown("""
            <h3 style='text-align: center; color: #427d44;'>🛠 How to use ContextHub?</h3>
            <br>
        """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>1. Select your Source</h4>
            <p>Choose between PDF, Image, Web, or Video tabs based on your needs.</p>
        </div><br>
        <div class="feature-card">
            <h4>2. Upload & Process</h4>
            <p>Click the process button to let Nomic, Groq & Gemini index your data.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>3. Ask Anything</h4>
            <p>Type your query in the chat box. The AI answers <b>only</b> based on your data.</p>
        </div><br>
        <div class="feature-card">
            <h4>4. Multi-Format Output</h4>
            <p>Get summaries, tables, or code extracted directly from your sources.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.caption("Developed by Varsha | Powered by Groq, Gemini, Pinecone & Nomic")

#########################################################################################################################

with tab1:
    st.header("PDF Analysis")

    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None
    if "pdf_answer" not in st.session_state:
        st.session_state.pdf_answer = None

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Upload & Preview")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            label_visibility="collapsed",
            key="pdf_input"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            uploaded_file.seek(0)

            if st.session_state.uploaded_filename != uploaded_file.name:
                st.session_state.file_processed = False
                st.session_state.pdf_answer = None

            if not st.session_state.file_processed:
                with st.spinner("Analyzing with AI..."):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf"
                            )
                        }
                        response = requests.post(
                            f"{BACKEND_URL}/upload",
                            files=files
                        )
                        if response.status_code == 200:
                            st.success(response.json()['message'])
                            st.session_state.file_processed = True
                            st.session_state.uploaded_filename = uploaded_file.name
                        else:
                            st.error(f"Backend Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")
            else:
                st.success(f"File '{uploaded_file.name}' already processed")

        else:
            st.info("Please select a PDF file first!")
            if st.session_state.file_processed:
                st.session_state.file_processed = False
                st.session_state.uploaded_filename = None
                st.session_state.pdf_answer = None

    with col2:
        st.subheader("Query Document")
        user_query = st.text_input("Type any question from the PDF...")
        ask_clicked = st.button("Ask pdf")

        if ask_clicked:
            if st.session_state.file_processed:
                if user_query:
                    with st.spinner("Thinking..."):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/query",
                                params={"query": user_query}
                            )
                            if response.status_code == 200:
                                st.session_state.pdf_answer = response.json().get("answer")
                            else:
                                st.error(f"Backend Error: {response.status_code}")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")
                else:
                    st.warning("Please type a question first!")
            else:
                st.error("Upload PDF first!")

        if st.session_state.pdf_answer:
            st.markdown("### Answer:")
            st.write(st.session_state.pdf_answer)

#########################################################################################################################

with tab2:
    st.header("🖼️ Image Intelligence")

    if "image_processed" not in st.session_state:
        st.session_state.image_processed = False
    if "uploaded_image_name" not in st.session_state:
        st.session_state.uploaded_image_name = None

    col1, col2 = st.columns([1,1], gap='large')

    with col1:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", width="content")

            if st.session_state.uploaded_image_name != uploaded_image.name:
                st.session_state.image_processed = False

            if not st.session_state.image_processed:
                with st.spinner("Processing image..."):
                    try:
                        files = {
                            "file": (
                                uploaded_image.name,
                                uploaded_image.getvalue(),
                                "image/png"
                            )
                        }
                        response = requests.post(
                            f"{BACKEND_URL}/process-image",
                            files=files
                        )
                        if response.status_code == 200:
                            st.session_state.image_processed = True
                            st.session_state.uploaded_image_name = uploaded_image.name
                            st.success("Image processed!")
                        else:
                            st.error("Backend error")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Upload an image to begin")

    with col2:
        query = st.text_input("Ask something about the image...")
        ask_clicked_image = st.button("Ask Image")

        if ask_clicked_image:
            if st.session_state.image_processed:
                if query:
                    with st.spinner("Analyzing with AI..."):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/query-image",
                                params={"query": query}
                            )
                            if response.status_code == 200:
                                st.write(response.json().get("answer"))
                            else:
                                st.error("Backend error")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Enter a question")
            else:
                st.error("Upload image first!")

#########################################################################################################################

with tab3:
    st.header("🌐 URL Intelligence")

    if "url_processed" not in st.session_state:
        st.session_state.url_processed = False
    if "current_url" not in st.session_state:
        st.session_state.current_url = None

    col1, col2 = st.columns([1,1], gap="large")

    with col1:
        url = st.text_input("Paste URL here")

        if url:
            if st.session_state.current_url != url:
                st.session_state.url_processed = False

            if not st.session_state.url_processed:
                with st.spinner("Fetching website..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/process-url",
                            params={"url": url}
                        )
                        if response.status_code == 200:
                            st.session_state.url_processed = True
                            st.session_state.current_url = url
                            st.success("URL processed!")
                        else:
                            st.error("Backend error")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Enter a URL to begin")

    with col2:
        query = st.text_input("Ask something about the website...")
        ask_clicked = st.button("Ask")

        if ask_clicked:
            if st.session_state.url_processed:
                if query:
                    with st.spinner("Analyzing..."):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/query-url",
                                params={"query": query}
                            )
                            if response.status_code == 200:
                                st.write(response.json().get("answer"))
                            else:
                                st.error("Backend error")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Enter a question")
            else:
                st.error("Process URL first!")

#########################################################################################################################

def get_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in url:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif "youtu.be" in url:
        return parsed_url.path.strip("/")
    return None

with tab4:
    st.header("🎥 YouTube Intelligence")

    # ---- SESSION STATE ----
    if "yt_processed" not in st.session_state:
        st.session_state.yt_processed = False
    if "yt_url" not in st.session_state:
        st.session_state.yt_url = None
    if "yt_input_key" not in st.session_state:
        st.session_state.yt_input_key = 0
    if "yt_answer" not in st.session_state:
        st.session_state.yt_answer = None
    if "yt_processing" not in st.session_state:
        st.session_state.yt_processing = False

    col1, col2 = st.columns([1, 1], gap="large")

    # ---------------- LEFT SIDE ----------------
    with col1:
        st.subheader("📎 Video Input")

        yt_url = st.text_input(
            "Paste YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            key=f"yt_input_{st.session_state.yt_input_key}"
        )

        if yt_url:
            video_id = get_video_id(yt_url)

            if video_id:
                col_img, col_clear = st.columns([8, 1])

                with col_clear:
                    if st.button("❌"):
                        st.session_state.yt_processed = False
                        st.session_state.yt_url = None
                        st.session_state.yt_input_key += 1
                        st.session_state.yt_answer = None
                        st.rerun()

                with col_img:
                    st.image(
                        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                        width=500
                    )

                process_clicked = st.button(
                    "🚀 Process Video",
                    disabled=st.session_state.yt_processing
                )

                if process_clicked:
                    st.session_state.yt_processing = True

                    with st.spinner("Fetching & indexing transcript..."):
                        try:
                            response = requests.post(
                                f"{BACKEND_URL}/process-youtube",
                                json={"url": yt_url}   # ✅ better practice
                            )

                            try:
                                data = response.json()
                            except:
                                st.error(f"❌ Invalid JSON response: {response.text}")
                                st.stop()

                            if response.status_code == 200:
                                if data.get("status") == "success":
                                    st.session_state.yt_processed = True
                                    st.session_state.yt_url = yt_url
                                    st.success("✅ Processed successfully!")
                                else:
                                    st.error(f"❌ {data.get('error', 'Something went wrong')}")
                            else:
                                st.error(f"❌ Server error: {response.status_code} | {response.text}")

                        except Exception as e:
                            st.error(f"Connection error: {e}")

                    st.session_state.yt_processing = False

                if st.session_state.yt_processed:
                    st.success("✅ Ready! Ask questions →")

            else:
                st.error("❌ Invalid YouTube URL")
        else:
            st.info("🔗 Paste a YouTube link to get started")

    # ---------------- RIGHT SIDE ----------------
    with col2:
        query = st.text_input("Ask something about the YouTube video...")
        ask_clicked = st.button("Ask the video")

        if ask_clicked:
            if not st.session_state.yt_processed:
                st.error("Process a YouTube URL first!")

            elif not query:
                st.warning("Enter a question")

            else:
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/query-youtube",
                            json={
                                "query": query,
                                "video_id": get_video_id(st.session_state.yt_url)  # ✅ FIX
                            }
                        )

                        if response.status_code == 200:
                            st.session_state.yt_answer = response.json().get("answer")
                        else:
                            st.error("Backend error")

                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.yt_answer:
            st.markdown("### Answer:")
            st.write(st.session_state.yt_answer)