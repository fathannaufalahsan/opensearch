import hashlib
from urllib.parse import urlparse
import plotly.express as px
import streamlit as st
import asyncio
import aiohttp
import pandas as pd
import logging
from bs4 import BeautifulSoup
from collections import defaultdict
import configparser
import os
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = configparser.ConfigParser()
if not config.read('config.ini'):
    logging.error("File config.ini tidak ditemukan.")
    st.stop()

API_KEYS = config['API_KEYS']
API_URLS = config['API_URLS']

class APIClient:
    """Class to handle API requests."""

    def __init__(self, api_keys, api_urls):
        self.api_keys = api_keys
        self.api_urls = api_urls
        self.cache = {}

    async def fetch_json_data(self, session, url, params=None, headers=None):
        try:
            if url in self.cache:
                logging.info(f"Using cached data for {url}")
                return self.cache[url]
            async with session.get(url, params=params, headers=headers, timeout=10) as response:
                response.raise_for_status()  # Ini akan memicu pengecualian untuk status 4xx dan 5xx
                data = await response.json()
                self.cache[url] = data
                return data
        except aiohttp.ClientResponseError as e:
            logging.error(f"Client error fetching data from {url}: {e.status} - {e.message}")
            return {}
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching data from {url}: {e}")
            return {}

    async def fetch_xml_data(self, session, url, params=None):
        try:
            if url in self.cache:
                logging.info(f"Using cached data for {url}")
                return self.cache[url]
            async with session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                data = BeautifulSoup(await response.text(), 'xml')
                self.cache[url] = data
                return data
        except aiohttp.ClientError as e:
            logging.error(f"Error fetching XML data from {url}: {e}")
            return None

    async def search_doaj(self, session, query, year=None):
        url = f"{self.api_urls['DOAJ_API']}?q={query}&page=1&pageSize=100"
        if year:
            url += f"&year={year}"
        data = await self.fetch_json_data(session, url)
        return [(r['bibjson'].get('title', 'No title'), r['link'][0].get('url', 'No link')) for r in data.get('results', [])]

    async def search_arxiv(self, session, query, year=None):
        url = self.api_urls['ARXIV_API']
        params = {'search_query': f'all:{query}', 'start': 0, 'max_results': 100}

        if year:
            params['search_query'] += f' AND submittedDate:[{year}0101 TO {year}1231]'  # Filter tahun dengan benar

        soup = await self.fetch_xml_data(session, url, params=params)

        return [(e.title.text, e.id.text) for e in soup.find_all('entry')] if soup else []

    async def search_core(self, session, query, year=None, count=10, start=0):
        """ Mencari artikel dari CORE API berdasarkan query dan tahun """
        url = self.api_urls.get("CORE_API", "https://api.core.ac.uk/v3/search/works/")
        headers = {
            "Accept": "application/json, application/xml"  # Menerima JSON & XML
        }
        params = {
            "q": query,
            "page": (start // count) + 1,  # CORE API menggunakan sistem halaman
            "limit": min(count, 50),
            "apiKey": self.api_keys.get("CORE_API_KEY", "")
        }
        if year:
            params["filter"] = f"year:{year}"

        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")

                    # Jika JSON
                    if "application/json" in content_type:
                        data = await response.json()

                    # Jika XML, parsing dengan BeautifulSoup
                    elif "application/xml" in content_type or "text/xml" in content_type:
                        xml_data = await response.text()
                        try:
                            data = self.parse_xml_to_dict(xml_data)  # Fungsi parsing XML
                        except Exception as xml_error:
                            logging.error(f"XML parsing error: {xml_error}")
                            return []

                    else:
                        logging.error(f"Unknown response format: {content_type}")
                        return []

                    # Ambil title, URL, dan sumber dari hasil pencarian
                    return [
                        (d.get("publisher", "Unknown Source"),
                         d.get("title", "No title"),
                         d.get("urls", ["No link"])[0] if isinstance(d.get("urls"), list) else d.get("urls", "No link"))
                        for d in data.get("results", [])
                    ]
                else:
                    logging.error(f"CORE API Error {response.status}: {await response.text()}")
                    return []

        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            return []

    def parse_xml_to_dict(self, xml_data):
        """ Mengubah XML menjadi dictionary yang kompatibel dengan JSON """
        soup = BeautifulSoup(xml_data, "xml")  # Menggunakan 'xml' parser

        results = []
        for entry in soup.find_all("result"):  # Pastikan tag ini sesuai dengan XML dari API
            title = entry.find("title").text if entry.find("title") else "No title"
            urls = [link.text for link in entry.find_all("url")]
            source = entry.find("publisher").text if entry.find("publisher") else "Unknown Source"

            results.append({
                "title": title,
                "urls": urls if urls else ["No link"],
                "publisher": source
            })

        return {"results": results}

    async def search_pubmed(self, session, query, year=None):
        url = self.api_urls['PUBMED_API']
        params = {'db': 'pubmed', 'term': query, 'retmax': 100, 'usehistory': 'y'}
        if year:
            params['term'] += f' AND {year}[PDAT]'
        soup = await self.fetch_xml_data(session, url, params=params)
        return [(f"PubMed Paper {id_tag.text}", f"https://pubmed.ncbi.nlm.nih.gov/{id_tag.text}/") for id_tag in
                soup.find_all('Id')] if soup else []

    async def search_springer(self, session, query, year=None):
        base_url = self.api_urls['SPRINGER_API'].split('?')[0]
        params = {
            'q': query,
            'rows': 50,
            'api_key': self.api_keys['SPRINGER_API_KEY']
        }
        if year:
            params['date'] = year
        data = await self.fetch_json_data(session, base_url, params=params)
        if data is None:
            return []
        return [(d.get('title', 'No title'), d.get('url', 'No link')) for d in data.get('records', [])]

    async def search_sciencedirect(self, session, query, year=None, count=25, start=0):
        url = self.api_urls.get('SCIENCE_DIRECT_API')
        headers = {
            "X-ELS-APIKey": self.api_keys.get("SCIDIRECT_API_KEY"),
            "Accept": "application/json"
        }
        params = {
            "query": query,
            "count": min(count, 50),
            "start": start
        }
        if year:
            params["date-range"] = f"{year}"

        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [(d.get('dc:title', 'No title'), d.get('prism:url', 'No link'))
                            for d in data.get('search-results', {}).get('entry', [])]
                else:
                    print(f"Error {response.status}: {await response.text()}")
                    return []
        except Exception as e:
            print(f"Exception occurred: {e}")
            return []

    async def search_jstor(self, session, query, year=None):
        url = self.api_urls['JSTOR_API']
        params = {'q': query}
        if year:
            params['fq'] = f'publication_date:{year}'
        data = await self.fetch_json_data(session, url, params=params)
        return [(d.get('title', 'No title'), d.get('url', 'No link')) for d in data.get('results', [])]

    async def search_openaire(self, session, query, year=None):
        """Cari publikasi di OpenAIRE berdasarkan judul dan tahun (opsional)."""
        url = "https://api.openaire.eu/search/publications"
        params = {"title": query}
        if year:
            params["date"] = str(year)

        try:
            logging.info(f"Mengirim permintaan ke {url} dengan parameter: {params}")  # Debugging URL

            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logging.error(f"Error {response.status}: Gagal mengambil data dari OpenAIRE.")
                    return []

                # Baca response sebagai teks XML
                xml_text = await response.text()

                # Parsing XML
                root = ET.fromstring(xml_text)

                output = []
                for item in root.findall(".//result"):  # Pastikan path ini sesuai dengan struktur XML
                    title = item.findtext("title", "No title")  # Sesuaikan dengan nama elemen XML
                    link = item.findtext("pid", "No link")
                    output.append((title, link))

                return output

        except Exception as e:
            logging.exception(f"Terjadi kesalahan saat mengakses OpenAIRE: {e}")
            return []

    async def search_openalex(self, session, query, year=None):
        url = f"{self.api_urls['OPENALEX_API']}?filter=title.search:{query}&per-page=100"
        if year:
            url += f"&filter=publication_year:{year}"
        data = await self.fetch_json_data(session, url)
        return [(d.get('title', 'No title'),
                 d.get('id', 'No link').replace('https://openalex.org/', 'https://openalex.org/works/')) for d in
                data.get('results', [])]

    async def search_doab(self, session, query, year=None):
        url = f"{self.api_urls['DOAB_API']}?query={query}&rows=100"
        if year:
            url += f"&year={year}"
        data = await self.fetch_json_data(session, url)
        # Pastikan untuk mengembalikan judul dan tautan yang sesuai
        return [(d.get('title', 'No title'), d.get('link', 'No link')) for d in data.get('results', [])]

    async def search_ieee_xplore(self, session, query, year=None):
        url = self.api_urls['IEEE_XPLORE_API']
        params = {'querytext': query, 'max_records': 100}
        if year:
            params['start_year'] = year
            params['end_year'] = year
        headers = {'X-API-Key': self.api_keys['IEEE_XPLORE_API_KEY']}
        data = await self.fetch_json_data(session, url, params=params, headers=headers)
        return [(d.get('title', 'No title'), d.get('html_url', 'No link')) for d in data.get('articles', [])]

    async def search(self, session, query, source, year=None):
        if source == 'DOAJ':
            return await self.search_doaj(session, query, year)
        elif source == 'arXiv':
            return await self.search_arxiv(session, query, year)
        elif source == 'CORE':
            return await self.search_core(session, query, year)
        elif source == 'PubMed':
            return await self.search_pubmed(session, query, year)
        elif source == 'Springer':
            return await self.search_springer(session, query, year)
        elif source == 'ScienceDirect':
            return await self.search_sciencedirect(session, query, year)
        elif source == 'JSTOR':
            return await self.search_jstor(session, query, year)
        elif source == 'OpenAIRE':
            return await self.search_openaire(session, query, year)
        elif source == 'OpenAlex':
            return await self.search_openalex(session, query, year)
        elif source == 'DOAB':
            return await self.search_doab(session, query, year)
        elif source == 'IEEE Xplore':
            return await self.search_ieee_xplore(session, query, year)
        else:
            logging.error(f"Unsupported source: {source}")
            return []

def show_notification(title, link):
    st.toast(f"New Journal Found: {title}\nRead it here: {link}", icon="🎉")

async def perform_search(query, api_client, year=None, timeout=10, max_retries=3):
    all_results = []
    search_functions = [
        ('DOAJ', api_client.search_doaj),
        ('arXiv', api_client.search_arxiv),
        ('CORE', api_client.search_core),
        ('PubMed', api_client.search_pubmed),
        ('Springer', api_client.search_springer),
        ('ScienceDirect', api_client.search_sciencedirect),
        ('JSTOR', api_client.search_jstor),
        ('OpenAIRE', api_client.search_openaire),
        ('OpenAlex', api_client.search_openalex),
        ('DOAB', api_client.search_doab),
        ('IEEE Xplore', api_client.search_ieee_xplore),
    ]

    # Inisialisasi status pencarian di Streamlit
    st.session_state.cancel_search = False
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_tasks = len(search_functions)
    completed_tasks = 0
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async def fetch_results(name, func):
            """Wrapper pencarian dengan retry dan timeout"""
            for attempt in range(1, max_retries + 1):
                try:
                    results = await asyncio.wait_for(func(session, query, year), timeout=timeout)
                    # 👉 Tambahkan nama sumber di sini jika belum ada di hasil
                    return [(name, title, link) for title, link in results]
                except asyncio.TimeoutError:
                    logging.warning(f"⚠️ Timeout ({attempt}/{max_retries}) di {name}")
                except Exception as e:
                    logging.error(f"❌ Error di {name} (Percobaan {attempt}): {e}")
                await asyncio.sleep(1.5)
            return []

        tasks = [fetch_results(name, func) for name, func in search_functions]

        for idx, future in enumerate(asyncio.as_completed(tasks), start=1):
            if st.session_state.cancel_search:
                logging.warning("❌ Pencarian dibatalkan oleh pengguna!")
                break

            results = await future
            if results:
                # ✅ Sekarang results sudah berisi (name, title, link)
                all_results.extend(results)

            # Update progress
            completed_tasks = idx
            elapsed_time = time.time() - start_time
            avg_time_per_task = elapsed_time / completed_tasks if completed_tasks else 1
            eta = avg_time_per_task * (total_tasks - completed_tasks)

            status_text.text(f"🔍 {completed_tasks}/{total_tasks} selesai. Perkiraan selesai: {eta:.1f} detik ⏳")
            progress_bar.progress(completed_tasks / total_tasks)

    status_text.text("✅ Pencarian selesai!")
    progress_bar.progress(1.0)

    return all_results

def save_results_to_csv(results, file_path):
    data = [[source, title, link] for source, title, link in results]
    pd.DataFrame(data, columns=["Source", "Title", "Link"]).to_csv(file_path, index=False)
    st.success(f"Results saved to {file_path}")

def save_results_to_excel(results, file_path):
    data = [[source, title, link] for source, title, link in results]
    pd.DataFrame(data, columns=["Source", "Title", "Link"]).to_excel(file_path, index=False)
    st.success(f"Results saved to {file_path}")

def normalize_source(source):
    """Normalisasi nama sumber: lowercase, hapus spasi, dan ambil domain jika URL."""
    source = source.strip().lower()
    parsed = urlparse(source)
    return parsed.netloc if parsed.netloc else source


def visualize_results(results, sumber_spesifik_list=None):
    """
    Visualisasikan jumlah hasil berdasarkan sumber, serta tampilkan jumlah untuk sumber spesifik tertentu.

    :param results: list of tuple (judul, sumber, ringkasan)
    :param sumber_spesifik_list: list sumber spesifik yang ingin dihitung (contoh: ['openalex', 'crossref'])
    """
    if not results:
        st.warning("⚠️ Tidak ada hasil untuk divisualisasikan.")
        return

    # Hitung jumlah per sumber
    sources = defaultdict(int)
    for _, source, _ in results:
        norm_source = normalize_source(source)
        sources[norm_source] += 1

    # Buat DataFrame
    df = pd.DataFrame(list(sources.items()), columns=['Source', 'Count'])

    # Tampilkan jumlah jurnal untuk sumber spesifik (jika diminta)
    if sumber_spesifik_list:
        st.subheader("📌 Jumlah Jurnal Berdasarkan Sumber Spesifik")
        for sumber in sumber_spesifik_list:
            sumber_norm = normalize_source(sumber)
            jumlah = sources.get(sumber_norm, 0)
            st.info(f"🔹 `{sumber}`: {jumlah} jurnal ditemukan")

    # Buat plot
    fig = px.bar(
        df,
        x='Source',
        y='Count',
        text='Count',
        color='Source',
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="📊 Jumlah Hasil per Sumber (Ternormalisasi)",
    )

    fig.update_layout(
        title_font_size=20,
        xaxis_title="Sumber",
        yaxis_title="Jumlah",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#333", size=14),
        showlegend=False,
        margin=dict(t=50, b=40, l=10, r=10)
    )
    fig.update_traces(textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

def load_preferences():
    if os.path.exists('preferences.ini'):
        config = configparser.ConfigParser()
        config.read('preferences.ini')
        if 'Preferences' in config:
            if 'source' in config['Preferences']:
                st.session_state.source = config['Preferences']['source']
            if 'year_filter' in config['Preferences']:
                st.session_state.year_filter = config['Preferences']['year_filter'] == 'True'
            if 'year' in config['Preferences']:
                st.session_state.year = int(config['Preferences']['year'])

def save_preferences():
    config = configparser.ConfigParser()
    config['Preferences'] = {
        'source': st.session_state.source,
        'year_filter': str(st.session_state.year_filter),
        'year': str(st.session_state.year)
    }
    with open('preferences.ini', 'w') as configfile:
        config.write(configfile)

def load_history():
    if os.path.exists('history.txt'):
        with open('history.txt', 'r') as file:
            history = file.read()
            st.session_state.history = history.split('\n')

def save_history():
    history = '\n'.join(st.session_state.history)
    with open('history.txt', 'w') as file:
        file.write(history)

def clear_history():
    st.session_state.history = []
    save_history()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'source' not in st.session_state:
    st.session_state.source = "All Sources"
if 'year_filter' not in st.session_state:
    st.session_state.year_filter = False
if 'year' not in st.session_state:
    st.session_state.year = 2023
if 'history' not in st.session_state:
    st.session_state.history = []

# Load preferences and history
load_preferences()
load_history()

# ========== MODERN LOGIN SYSTEM ==========
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Layout tergantung status login
if st.session_state.authenticated:
    st.set_page_config(
        page_title="OpenSearch 2025",
        page_icon="OpenSearch.ico",
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    st.set_page_config(
        page_title="Login",
        page_icon="OpenSearch.ico",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

# FORCE DARK MODE (CSS Manual)
dark_mode_css = """
<style>
    body, .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    section[data-testid="stSidebar"] {
        background-color: #16191f;
    }
    div[data-baseweb="input"] input,
    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
    }
</style>
"""
st.markdown(dark_mode_css, unsafe_allow_html=True)

# Fungsi untuk hashing password menggunakan SHA-256
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Dummy akun dengan password sudah di-hash
accounts = {
    "admin": {"password": hash_password("admin123"), "role": "Admin"},
    "demo": {"password": hash_password("demo"), "role": "Demo"},
}

# Inisialisasi state
for key in ["authenticated", "username", "role", "show_password"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "authenticated" else ""

def login_ui():
    st.markdown("""
        <style>
            /* Animasi Glow */
            @keyframes glow {
                0% { text-shadow: 0 0 5px #33ccff, 0 0 10px #33ccff, 0 0 15px #33ccff; }
                50% { text-shadow: 0 0 10px #00e6e6, 0 0 20px #00e6e6, 0 0 30px #00e6e6; }
                100% { text-shadow: 0 0 5px #33ccff, 0 0 10px #33ccff, 0 0 15px #33ccff; }
            }

            h4 {
                color: #00FFFF;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin-top: 0;
                animation: glow 1.5s ease-in-out infinite;
            }
        </style>
        <div style="text-align: center;">
            <h2 style="color: #2196F3; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin-bottom: 0.2em;">
                🔐 Welcome Back!
            </h2>
            <h4>
               🔎 OpenSearch 2025
            </h4>
            <p style="color: #CCCCCC; font-size: 16px; font-style: italic;">
                Please login using your credentials to access your dashboard 🚀
            </p>
        </div>
    """, unsafe_allow_html=True)


    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        submitted = st.form_submit_button("🔓 Login")

    if submitted:
        if username in accounts and accounts[username]["password"] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.role = accounts[username]["role"]
            st.success(f"✅ Welcome, {username}!")
            st.balloons()
            try:
                st.rerun()
            except AttributeError:
                st.info("🔄 Please refresh manually (F5)")
                st.stop()
        else:
            st.error("❌ Incorrect username or password.")

# ✅ Login Gate
if not st.session_state.authenticated:
    login_ui()
    st.stop()
# ======== END LOGIN =========

# Inisialisasi state
# 🚨 Ini harus jadi baris Streamlit pertama
st.title("🔎 OpenSearch 2025")
st.markdown("Temukan jurnal dari berbagai sumber terbuka dengan cepat dan mudah 🚀")

# Inisialisasi state
for key, default in {
    "results": [], "history": [], "year_filter": False, "year": 2023
}.items():
    st.session_state.setdefault(key, default)

# ================== 🔍 PENCARIAN ==================
with st.container():
    st.header("🔍 Pencarian Jurnal")
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Masukkan Kata Kunci", placeholder="Contoh: machine learning")
    with col2:
        source = st.selectbox("Pilih Sumber", [
            "All Sources", 'DOAJ', 'arXiv', 'CORE', 'PubMed', 'Springer',
            'ScienceDirect', 'JSTOR', 'OpenAIRE', 'OpenAlex', 'DOAB', 'IEEE Xplore'
        ])

    st.markdown("##### 🎯 Opsi Tambahan")
    with st.expander("⚙️ Filter Tahun"):
        year_filter = st.checkbox("Aktifkan Filter Tahun", value=st.session_state.year_filter)
        year = st.number_input("Tahun Publikasi", 2000, 2025, value=st.session_state.year,
                               disabled=not year_filter)

    if st.button("🔍 Mulai Pencarian"):
        if not query:
            st.warning("Silakan masukkan kata kunci pencarian.")
        else:
            with st.spinner("Sedang mencari data..."):
                api_client = APIClient(API_KEYS, API_URLS)
                results = asyncio.run(perform_search(query, api_client, year if year_filter else None))
                st.session_state.results = results
                entry = f"Query: {query}, Year: {year if year_filter else 'All'}"
                st.session_state.history.append(entry)
                save_history()
                save_preferences()
            st.success(f"✅ Ditemukan {len(results)} hasil.")

# ================== 📄 HASIL PENCARIAN ==================
# Filter hasil pencarian berdasarkan sumber
filtered_results = (
    [r for r in st.session_state.results if r[0] == source]
    if source != "All Sources"
    else st.session_state.results
)

if filtered_results:
    st.markdown("""
        <div>
            <h2 style="
                color: white;
                font-weight: 700;
                font-size: 1.75rem;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            ">
                📄 Hasil Pencarian
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # Buat DataFrame
    df = pd.DataFrame(filtered_results, columns=["Sumber", "Judul", "Tautan"])

    # Tampilkan tabel data
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Ekspander notifikasi hasil baru
    with st.expander("Hasil Detail", expanded=False):
        for _, title, link in filtered_results:
            st.markdown(f"""
                <div style="
                    background-color: #f9f9f9;
                    border-left: 4px solid #4A90E2;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
                ">
                    <h4 style="
                        margin: 0 0 0.5rem 0;
                        color: #222;
                        font-weight: 600;
                        font-size: 1.1rem;
                    ">
                        {title}
                    </h4>
                    <a href="{link}" target="_blank" style="
                        color: #4A90E2;
                        font-weight: 500;
                        text-decoration: none;
                        font-size: 0.95rem;
                    ">
                        🔗 Buka Tautan
                    </a>
                </div>
            """, unsafe_allow_html=True)

# ================== 📤 EKSPOR ==================
st.markdown("---")
st.header("📤 Ekspor Data")
col_csv, col_excel = st.columns(2)

with col_csv:
    st.markdown("### 💾 Simpan CSV")
    if filtered_results:
        filename_csv = st.text_input("Nama File CSV", "results.csv")
        if st.button("Simpan CSV"):
            save_results_to_csv(filtered_results, filename_csv)
    else:
        st.info("Belum ada data untuk disimpan.")

with col_excel:
    st.markdown("### 📊 Simpan Excel")
    if filtered_results:
        filename_excel = st.text_input("Nama File Excel", "results.xlsx")
        if st.button("Simpan Excel"):
            save_results_to_excel(filtered_results, filename_excel)
    else:
        st.info("Belum ada data untuk disimpan.")

# ================== 📈 VISUALISASI ==================
# Jika ada hasil dan user klik tombol visualisasi
if filtered_results:
    # Custom tombol style dengan HTML & CSS
    st.markdown("""
        <style>
        .visual-btn {
            display: inline-block;
            background-color: #4A90E2;
            color: white;
            padding: 0.75rem 1.25rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: background 0.3s ease;
            margin-bottom: 1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        .visual-btn:hover {
            background-color: #3a7bd5;
        }
        </style>
    """, unsafe_allow_html=True)

    # Buat tombol menggunakan HTML, lalu trigger pakai st.button biasa
    if st.button("📈 Visualisasi Hasil", key="vis_btn"):
        visualize_results(filtered_results)

# ================== TENTANG ==================
with st.expander("ℹ️ Tentang OpenSearch"):
    st.markdown("""
    ### OpenSearch 2025  
    - **Dikembangkan oleh**: Fathan Naufal Ahsan 
    - **Brand**: Ahsan Karya  
    - **Versi**: `v25.6.0`  
    - **Deskripsi**: Pencarian jurnal akses terbuka dari sumber global  
    - **Kontak**: fathannaufalahsan.18@gmail.com  
    """)

# ================== 📜 RIWAYAT ==================
st.markdown("---")
st.header("📜 Riwayat Pencarian")

if st.session_state.history:
    with st.expander("🧾 Lihat Riwayat"):
        history_text = "\n".join(st.session_state.history)
        st.text_area("Riwayat Pencarian", value=history_text, height=150, disabled=True)
        if st.button("🧹 Bersihkan Riwayat"):
            clear_history()
            st.success("🗑️ Riwayat berhasil dihapus.")
else:
    st.info("Belum ada pencarian sebelumnya.")

# --- Tombol Logout ---
st.markdown("---")
if st.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.role = ""
    try:
        st.rerun()
    except AttributeError:
        st.warning("🔄 Please refresh manually (F5).")
        st.stop()
