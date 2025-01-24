# LCVBOT3.9Indo.py
# Created/Modified files during execution: ["LCVBOT3.9Indo.py"]

import os
import streamlit as st
import torch
import toml

# LangChain & Komponen Pendukung
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
# ─────────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURASI & HALAMAN
# ─────────────────────────────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="LCV-ASISSTANT",
    page_icon="🤖",
    layout="wide"
)

# Tampilkan Judul
st.title("🤖 LCV ASSISTANT")
st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
    <p style='font-size: 18px; margin: 0;'>
    <strong>SELAMAT DATANG !</strong> - Chatbot ini adalah Asisten yang dilatih untuk <strong>membantu AoC dalam implementasi LCV AKHLAK 2025.</strong>
    Pastikanlah Anda memiliki koneksi internet yang baik dan stabil. Terimakasih atas kesabarannya menunggu chatbot siap untuk digunakan 🙏
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────────
# 2. SIDEBAR PUSTAKA
# (Mengambil keseluruhan data pustaka dari LCVBOT3.8Indo_D)
# ─────────────────────────────────────────────────────────────────────────────────

PUSTAKA_DATA = {
    "files": [
        {
            "title": "9 Parameter 2025",
            "description": "9 Parameter Penilaian LCV AKHLAK 2025",
            "url": "https://drive.google.com/file/d/11DOL9kG0ttp0ilJ_5ykLrSjKVEI9IJlV/view?usp=sharing"
        },
        {
            "title": "10 Fokus Keberlanjutan Pertamina",
            "description": "Penjelasan 10 Fokus Keberlanjutan Pertamina beserta contohnya.",
            "url": "https://drive.google.com/file/d/1FTIttFp17nGh5Pfc_w-wS-Xf7D_aLUrg/view?usp=sharing"
        },
        {
            "title": "Contoh PCB dengan Allignment terhadap 10 Fokus Keberlanjutan Pertamina",
            "description": "Berbagai contoh PCB yang memiliki program budaya menyasar pada 10 Fokus Keberlnajutan Pertamina.",
            "url": "https://drive.google.com/file/d/17Bx_ha1o01UsrovI6TVadmupP9LxN-J5/view?usp=sharing"
        },
        {
            "title": "Contoh Klasifikasi Program",
            "description": "Contoh-contoh klasifikasi program: Strategis, Taktikal, Operasional",
            "url": "https://docs.google.com/spreadsheets/d/1irDS2zSD8yavfEf5uLDSpuCY65T_UIe0/edit?usp=sharing"
        },
        {
            "title": "Form Kuantifikasi Impact to Business",
            "description": "Form kuantifikasi impact to business dan contoh pengisian",
            "url": "https://docs.google.com/spreadsheets/d/1W2jlrIhiJac_1oLd86dSSXLMihgV1KO4/edit?usp=sharing"
        },
        {
            "title": "Sosialisasi LCV 2025",
            "description": "Materi sosialisasi LCV 2025",
            "url": "https://drive.google.com/file/d/1iXtwOtd0BCF4tQnz2R2Obek3I2E0h5cM/view?usp=sharing"
        },
        {
            "title": "Dashboard PowerBI",
            "description": "Nilai Kualitatif Evidence Bulanan",
            "url": "https://ptm.id/skorlivingcorevaluesAKHLAK"
        },
        {
            "title": "Konfirmasi Evidence",
            "description": "Form Konfirmasi Evidence LCV 2025 dibuka setiap tanggal 26-27",
            "url": "https://ptm.id/FormKonfirmasiEvidenceLCV2025"
        }
    ]
}

def display_pustaka():
    st.sidebar.title("Referensi Penunjang")
    st.sidebar.write("Daftar Referensi yang Tersedia:")
    for file in PUSTAKA_DATA["files"]:
        st.sidebar.write(f"**{file['title']}**")
        st.sidebar.write(f"Deskripsi: {file['description']}")
        st.sidebar.markdown(f"[Buka Link]({file['url']})")
        st.sidebar.write("---")

display_pustaka()

# ─────────────────────────────────────────────────────────────────────────────────
# 3. CUSTOM PROMPT
# (Gabungan Template Indonesia & pendekatan ringkas)
# ─────────────────────────────────────────────────────────────────────────────────

template = """Anda adalah asisten AI yang ahli dalam dokumen berbahasa Indonesia.
Gunakan konteks berikut untuk menjawab pertanyaan dengan:
1. Bahasa Indonesia yang baik, formal, dan terstruktur
2. Referensi ke bagian dokumen yang relevan jika ada
3. Penjelasan yang jelas dan terorganisir

Konteks: {context}
Riwayat Chat: {chat_history}
Pertanyaan: {question}

Instruksi khusus:
- Jika informasi tidak ditemukan dalam konteks, katakan "Mohon maaf, saya tidak memiliki informasi tersebut dalam dokumen yang tersedia."
- Berikan jawaban yang ringkas namun komprehensif
- Jika ada istilah teknis, berikan penjelasan singkatnya

Jawaban:"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

# ─────────────────────────────────────────────────────────────────────────────────
# 4. PREPROCESS DOCUMENT (FUNGSIONALITAS UTK BAHASA INDONESIA)
# ─────────────────────────────────────────────────────────────────────────────────

def preprocess_document(text: str) -> str:
    """Preprocessing khusus untuk dokumen Bahasa Indonesia."""
    import re

    # Bersihkan karakter khusus
    text = re.sub(r'[^\w\s\.]', ' ', text)

    # Normalisasi spasi
    text = ' '.join(text.split())

    # Handling untuk singkatan umum bahasa Indonesia
    abbreviations = {
        'yg': 'yang',
        'dgn': 'dengan',
        'utk': 'untuk',
        'tsb': 'tersebut',
        'dll': 'dan lain-lain',
        'dst': 'dan seterusnya',
        'dsb': 'dan sebagainya',
        'spt': 'seperti',
        'krn': 'karena',
        'pd': 'pada',
        'dr': 'dari',
        'knp': 'kenapa',
        'LCV': 'Living Core Values',
        'PCB': 'Project Charter Budaya'
    }
    for abbr, full in abbreviations.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)

    return text

# ─────────────────────────────────────────────────────────────────────────────────
# 5. LOAD DOKUMEN & BANGUN VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────────

DOCUMENTS_PATH = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def process_documents():
    """
    Memuat semua dokumen PDF dalam folder 'documents',
    melakukan preprocessing text, lalu split jadi chunk,
    membuat FAISS vector store dengan embedding bahasa Indonesia.
    """
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        st.warning(f"Folder {DOCUMENTS_PATH} telah dibuat. Silakan tambahkan file PDF Anda.")
        return None

    pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        st.warning("Tidak ada file PDF yang ditemukan dalam folder 'documents'.")
        return None

    all_documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(DOCUMENTS_PATH, pdf_file))
        docs = loader.load()

        # Lakukan preprocess pada setiap chunk dokumen
        for doc in docs:
            doc.page_content = preprocess_document(doc.page_content)
        all_documents.extend(docs)

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    doc_chunks = text_splitter.split_documents(all_documents)

    # Embedding
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name="LazarusNLP/all-indo-e5-small-v4",
        model_kwargs={'device': device}
    )

    # Buat Vector Store
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    return vector_store

# ─────────────────────────────────────────────────────────────────────────────────
# 6. MEMBANGUN CONVERSATION CHAIN
# ─────────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "gpt-3.5-turbo"  
TEMPERATURE = 0.45
MAX_TOKENS = 1800

def load_api_key():
    """
    Membaca OPENAI_API_KEY dari environment
    atau dari .streamlit/secrets.toml (opsional).
    """
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key

        with open('.streamlit/secrets.toml', 'r') as f:
            config = toml.load(f)
            return config.get('OPENAI_API_KEY')
    except Exception as e:
        st.error(f"Gagal memuat OpenAI API Key: {str(e)}")
        return None

def get_conversation_chain(vector_store, api_key):
    """
    Membuat ConversationalRetrievalChain menggunakan model ChatOpenAI.
    Memori percakapan disimpan dengan ConversationBufferMemory.
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        presence_penalty=0.1,
        frequency_penalty=0.1,
        top_p=0.95
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': CUSTOM_PROMPT}
    )
    return chain

# ─────────────────────────────────────────────────────────────────────────────────
# 7. MAIN APLIKASI
# (Inisialisasi + UI Chat)
# ─────────────────────────────────────────────────────────────────────────────────

def main():
    api_key = load_api_key()
    if not api_key:
        st.error("API key tidak ditemukan. Silakan set OPENAI_API_KEY di environment atau .streamlit/secrets.toml.")
        return

    if "vector_store" not in st.session_state:
        with st.spinner("Memproses dokumen..."):
            vector_store = process_documents()
            if vector_store is None:
                return
            st.session_state.vector_store = vector_store

    if "conversation" not in st.session_state:
        if "vector_store" in st.session_state:
            st.session_state.conversation = get_conversation_chain(st.session_state.vector_store, api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan riwayat percakapan
    for msg in st.session_state.messages:
        role_class = "user-message" if msg["role"] == "user" else "assistant-message"
        with st.chat_message(msg["role"]):
            st.markdown(f"<div class='chat-message {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    prompt = st.chat_input("✍️ Tulis pertanyaan Anda di sini. Fokus pada topik implementasi LCV AKHLAK 2025:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"<div class='chat-message user-message'>{prompt}</div>", unsafe_allow_html=True)

        with st.chat_message("assistant"):
            try:
                with st.spinner("🤔 Memproses..."):
                    # Pertanyaan diteruskan ke chain
                    response = st.session_state.conversation({"question": prompt})
                    answer = response.get('answer', 'Maaf, saya tidak menemukan jawaban.')
                    st.markdown(f"<div class='chat-message assistant-message'>{answer}</div>", unsafe_allow_html=True)
                    # Simpan ke riwayat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_message = f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# ─────────────────────────────────────────────────────────────────────────────────
# 8. DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────────

st.markdown("""
---
**Disclaimer:**
- Sistem ini menggunakan AI-LLM dan dapat menghasilkan jawaban yang tidak selalu akurat.
- LCV assistant tidak merespons pertanyaan di luar konteks implementasi LCV AKHLAK.
- Terkait **9 Parameter LCV 2025**, silahkan mengacu pada dokumen di side bar Referensi Penunjang.
- Untuk mengurangi halusinasi dan meningkatkan akurasi, gunakanlah sistem prompt **R-A-G**, contoh:
  • **[Role]** Anda adalah seorang ahli Living Core Values AKHLAK yang kreatif.
  • **[Action]** Tolong berikan 3 buah ide program ONE KOLAB yang strategis,
  • **[Goals]** dengan tujuan menciptakan safety dan produktivitas yang lebih baik.
- Mohon verifikasi informasi penting dengan sumber terpercaya.
""")

# Jalankan Aplikasi
if __name__ == "__main__":
    main()
