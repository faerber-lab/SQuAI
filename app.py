import sys
import json

try:
    import streamlit as st
    import requests
    import time
    import os
    import subprocess
    import configparser
    import re
except ModuleNotFoundError as e:
    print(f"Module not found: {e}. Did you run this script via frontend.sh?")
    sys.exit(1)

AVAILABLE_MODELS = [
    "google/gemma-4-31B-it",
    "moonshotai/Kimi-K2.6",
    "openai/gpt-oss-120b",
    "meta-llama/Llama-3.3-70B-Instruct",
]

def get_script_path():
    try:
        script_path = os.path.abspath(sys.argv[0])
        return script_path
    except Exception as e:
        print("Error detecting the path:", str(e))
        return None

def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    except Exception as e:
        print("Error while determining script directory:", str(e))
        return None

def read_hpc_config():
    """Read HPC credentials from defaults.ini"""
    try:
        script_dir = get_script_dir()
        if script_dir is None:
            return None, None
        
        config_path = os.path.join(script_dir, "defaults.ini")
        if not os.path.isfile(config_path):
            print(f"Config file not found: {config_path}")
            return None, None
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        username = config.get('DEFAULT', 'username', fallback=None)
        partition = config.get('DEFAULT', 'partition', fallback=None)
        
        return username, partition
    except Exception as e:
        print(f"Error reading config: {e}")
        return None, None

def check_hpc_connection(username):
    """Check if SSH connection to HPC is possible"""
    if not username:
        return False, "No username configured"
    
    host = f"{username}@login1.capella.hpc.tu-dresden.de"
    
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, "echo", "connected"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0 and "connected" in result.stdout:
            return True, "SSH connection successful"
        else:
            return False, f"SSH connection failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "SSH connection timeout"
    except Exception as e:
        return False, f"SSH error: {str(e)}"

def get_job_status(username):
    """Get SLURM job status from HPC"""
    if not username:
        return None, "No username configured"
    
    host = f"{username}@login1.capella.hpc.tu-dresden.de"
    
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, "squeue", "--me"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode != 0:
            return None, f"squeue command failed: {result.stderr}"
        
        output = result.stdout
        lines = output.strip().split('\n')
        
        if len(lines) <= 1:
            return None, "No jobs found in queue"
        
        # Parse job information
        jobs = []
        for line in lines[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 6:
                job = {
                    'jobid': parts[0],
                    'partition': parts[1],
                    'name': parts[2],
                    'user': parts[3],
                    'state': parts[4],
                    'time': parts[5]
                }
                jobs.append(job)
        
        return jobs, None
    except subprocess.TimeoutExpired:
        return None, "Command timeout"
    except Exception as e:
        return None, f"Error: {str(e)}"

def get_pending_reason(username, jobid):
    """Get reason why job is pending using whypending"""
    if not username or not jobid:
        return None
    
    host = f"{username}@login1.capella.hpc.tu-dresden.de"
    
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", host, "whypending", jobid],
            capture_output=True,
            text=True,
            timeout=20
        )
        
        if result.returncode != 0:
            return None
        
        output = result.stdout
        
        # Extract relevant information
        info = {
            'full_output': output,
            'reason': None,
            'position': None,
            'fairshare': None,
            'estimated_start': None
        }
        
        # Parse reason
        reason_match = re.search(r'Reason (\w+)', output)
        if reason_match:
            info['reason'] = reason_match.group(1)
        
        # Parse position
        position_match = re.search(r'Position in queue: (\d+)', output)
        if position_match:
            info['position'] = position_match.group(1)
        
        # Parse fairshare
        fairshare_match = re.search(r'FairShare rating is low ([\d.]+)', output)
        if fairshare_match:
            info['fairshare'] = fairshare_match.group(1)
        
        # Parse estimated start
        estimated_match = re.search(r'Estimated start time: (.+)', output)
        if estimated_match:
            info['estimated_start'] = estimated_match.group(1)
        
        return info
    except Exception as e:
        print(f"Error getting pending reason: {e}")
        return None

def show_503_page():
    """Display custom 503 error page with HPC status"""
    st.markdown(
        """
        <div style="
            border: 3px solid #ff4444;
            border-radius: 12px;
            padding: 24px;
            background-color: #2a1a1a;
            color: #f5f5f5;
            margin: 20px 0;
        ">
            <h2 style="color: #ff6666; margin-top: 0;">⚠️ Service Currently Unavailable</h2>
            <p style="font-size: 1.1em;">The SQuAI service on HPC is currently not running.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    username, partition = read_hpc_config()
    
    if not username:
        st.error("❌ Could not read HPC configuration from defaults.ini")
        return
    
    with st.spinner("Checking HPC connection..."):
        ssh_ok, ssh_msg = check_hpc_connection(username)
    
    if not ssh_ok:
        st.markdown(
            f"""
            <div style="
                border: 2px solid #ff8800;
                border-radius: 8px;
                padding: 16px;
                background-color: #2a2010;
                color: #ffcc88;
                margin: 10px 0;
            ">
                <strong>🔌 SSH Connection Failed</strong><br>
                {ssh_msg}
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    st.success("✅ SSH connection to HPC successful")
    
    with st.spinner("Retrieving job status..."):
        jobs, error = get_job_status(username)
    
    if error:
        st.error(f"❌ Error retrieving job status: {error}")
        return
    
    if not jobs:
        st.warning("⚠️ No jobs found in the queue. The service might need to be started manually.")
        return
    
    # Display job status
    for job in jobs:
        state = job['state']
        jobid = job['jobid']
        
        if state == 'R':  # Running
            st.markdown(
                f"""
                <div style="
                    border: 2px solid #44ff44;
                    border-radius: 8px;
                    padding: 16px;
                    background-color: #1a2a1a;
                    color: #ccffcc;
                    margin: 10px 0;
                ">
                    <strong>🟢 Job is Running</strong><br>
                    Job ID: <code>{jobid}</code><br>
                    The service is currently running but may need a moment to become fully available.<br>
                    Please try again in a few moments.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        elif state == 'PD':  # Pending
            st.markdown(
                f"""
                <div style="
                    border: 2px solid #ffaa44;
                    border-radius: 8px;
                    padding: 16px;
                    background-color: #2a2510;
                    color: #ffddaa;
                    margin: 10px 0;
                ">
                    <strong>🟡 Job is Pending</strong><br>
                    Job ID: <code>{jobid}</code><br>
                    The job is waiting in the queue to start.
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Get detailed pending information
            with st.spinner("Analyzing queue position..."):
                pending_info = get_pending_reason(username, jobid)
            
            if pending_info:
                details = []
                
                if pending_info['reason']:
                    reason_text = pending_info['reason']
                    if reason_text == "Priority":
                        reason_text += " (Job will start as soon as resources free up)"
                    details.append(f"**Reason:** {reason_text}")
                
                if pending_info['position']:
                    details.append(f"**Position in queue:** {pending_info['position']}")
                
                if pending_info['fairshare']:
                    details.append(f"**FairShare rating:** {pending_info['fairshare']}")
                
                if pending_info['estimated_start']:
                    details.append(f"**Estimated start:** {pending_info['estimated_start']}")
                
                if details:
                    st.markdown("**Queue Details:**")
                    for detail in details:
                        st.markdown(f"- {detail}")
                
                with st.expander("Show full queue analysis"):
                    st.code(pending_info['full_output'])
        
        else:
            st.markdown(
                f"""
                <div style="
                    border: 2px solid #888888;
                    border-radius: 8px;
                    padding: 16px;
                    background-color: #222222;
                    color: #cccccc;
                    margin: 10px 0;
                ">
                    <strong>ℹ️ Job Status: {state}</strong><br>
                    Job ID: <code>{jobid}</code>
                </div>
                """,
                unsafe_allow_html=True
            )

def start_backend():
    script_dir = get_script_dir()
    if script_dir is None:
        return False

    shell_script = os.path.join(script_dir, "start_backend_from_enterprise_cloud.sh")

    if not os.path.isfile(shell_script):
        print("Error: Script not found:", shell_script)
        return False

    try:
        process = subprocess.Popen(
            ["bash", shell_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setpgrp
        )
        print("Backend started. PID:", process.pid)
        return True
    except Exception as e:
        print("Error while starting backend:", str(e))
        return False

def wait_for_backend(url, timeout=60, wait_between=2):
    """Wait until the backend endpoint becomes available."""
    start_time = time.time()
    while True:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("Backend is available.")
                return True
        except Exception:
            pass

        if time.time() - start_time > timeout:
            print(f"Backend not reachable after {timeout} seconds.")
            return False

        time.sleep(wait_between)

def check_backend_available(url, timeout=5):
    """Check if backend is available without retries"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=60, show_spinner=False)
def check_external_api_health():
    """Checks the ScaDS AI LLM health endpoint and identifies specific unhealthy models.

    Cached for 60s so the 5s-timeout HTTPS call does NOT run on every Streamlit rerun
    (which happens on every chat message) — only once per minute."""
    url = "https://llm.scads.ai/health"
    try:
        api_key = (
            os.environ.get("SCADS_API_KEY")
            or (open(os.path.expanduser("/etc/scads_api_key")).read().strip() if os.path.exists(os.path.expanduser("/etc/scads_api_key")) else None)
            or (open("/etc/scads_agent_api_key").read().strip() if os.path.exists("/etc/etc/scads_agent_api_key") else None)
        )
        
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            unhealthy_count = data.get("unhealthy_count", 0)
            
            if unhealthy_count == 0:
                return "green", "All systems operational"
            else:
                # Extract model names from the unhealthy_endpoints list
                unhealthy_list = data.get("unhealthy_endpoints", [])
                model_names = {item.get("model", "Unknown Model") for item in unhealthy_list}
                
                # Only flag models that are in AVAILABLE_MODELS
                affected_models = model_names & set(AVAILABLE_MODELS)
                
                if not affected_models:
                    return "green", "All relevant systems seem operational"
                
                names_str = ", ".join(affected_models)
                affected_count = len(affected_models)
                
                return "yellow", f"Warning: {affected_count} unhealthy endpoint(s) affecting available models: {names_str}"
        
        return "red", f"API Error: {response.status_code}"
    except Exception as e:
        return "red", f"Connection failed: {str(e)}"

def post_with_retry(url, payload, wait_between=30, max_retries=5, max_backend_restarts=5):
    backend_restarts = 0

    while backend_restarts < max_backend_restarts:
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=30)
                
                # Check for 503 Service Unavailable
                if response.status_code == 503:
                    return {'status_code': 503, 'is_503': True}
                
                response.raise_for_status()
                return response
            except requests.exceptions.ConnectionError:
                # Backend is not reachable at all - return 503
                print(f"POST attempt {attempt+1}/{max_retries}: Connection refused")
                if attempt == 0:  # On first attempt, immediately return 503
                    return {'status_code': 503, 'is_503': True}
                time.sleep(wait_between)
            except requests.exceptions.Timeout:
                print(f"POST attempt {attempt+1}/{max_retries}: Request timeout")
                time.sleep(wait_between)
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 503:
                    return {'status_code': 503, 'is_503': True}
                print(f"POST attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(wait_between)
            except Exception as e:
                print(f"POST attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(wait_between)

        print(f"All {max_retries} attempts failed. Starting backend...")
        if start_backend():
            backend_restarts += 1
            if not wait_for_backend("http://localhost:8000", timeout=600):
                print("Backend did not become available in time.")
                continue
        else:
            print("Backend could not be started.")
            backend_restarts += 1

    raise RuntimeError(f"POST failed after {max_backend_restarts} backend restarts")

# Page Configuration
status_color, status_message = check_external_api_health()
st.set_page_config(page_title="SQuAI", layout="wide")
st.markdown("""
<style>
/* ── Smooth transitions on ALL interactive elements ── */
button, input, select, textarea,
.stButton > button,
.stTextInput > div > div > input,
.stSelectbox > div > div,
[data-testid="stExpander"],
.stSlider > div {
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* ── Buttons: lift on hover ── */
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255, 255, 255, 0.08);
}
.stButton > button:active {
    transform: translateY(0px);
}
</style>
""", unsafe_allow_html=True)
st.title("SQuAI")

# First, ensure you have the logic to define these variables before the markdown call
# status_color, status_message = check_external_api_health() 

st.markdown(f"""
<style>
/* Footer fixieren */
.footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #111;
    color: #aaa;
    text-align: center;
    padding: 10px;
    font-size: 0.85em;
    z-index: 100;
    border-top: 1px solid #444;
}}

.footer a {{
    color: #aaa;
    text-decoration: none;
    margin: 0 15px;
}}

.footer a:hover {{
    text-decoration: underline;
}}
</style>

<div class="footer" title="{status_message}">
    <a href="https://scads.ai/imprint/" target="_blank">Impressum</a>
    <a href="https://scads.ai/privacy/" target="_blank">Datenschutzerklärung</a>
    <a href="https://scads.ai/accessibility/" target="_blank">Barrierefreiheit</a>
    <span style="
        height: 12px;
        width: 12px;
        background-color: {status_color};
        border-radius: 50%;
        display: inline-block;
        margin-left: 10px;
        margin-right: 5px;
        vertical-align: middle;
        box-shadow: 0 0 5px {status_color};
    "></span>
    <span style="font-size: 0.9em; color: #aaa; vertical-align: middle;">API Status</span>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.markdown("## Settings")

model_choice = st.sidebar.selectbox("Language Model", AVAILABLE_MODELS, index=0)
retrieval_choice = st.sidebar.selectbox("Retrieval Model", ["Hybrid","bm25", "e5"], index=0)

with st.sidebar.expander("⚙️ Advanced"):
    n_value = st.slider(
        "N_VALUE",
        0.0, 1.0, 0.5, step=0.01,
        help="Controls document filtering stringency: if 0, filtering is very strict; if 1, it’s very tolerant."
    )
    top_k = st.number_input(
        "TOP_K",
        min_value=1, max_value=20, value=5, step=1,
        help="Determines how many of the top-ranked documents are considered for answer generation."
    )
    alpha = st.slider(
        "ALPHA",
        0.0, 1.0, 0.65, step=0.01,
        help="A weighting factor that balances the influence between E5 and BM25 retrieval methods, calculated as Alpha × E5 + (1 − Alpha) × BM25."
    )

# Check backend availability on page load (only once per session)
#if 'backend_check_done' not in st.session_state:
#    st.session_state.backend_check_done = True
#    if not check_backend_available("http://localhost:8000"):
#        show_503_page()
#        st.stop()

# ─────────────────────────── Chat interface ───────────────────────────
# Conversation history persists across reruns. Each assistant turn keeps its
# references + debug_info so it re-renders fully on replay.
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role", "content", "references", "debug_info"}]

if st.sidebar.button("🗑️ New chat"):
    st.session_state.messages = []
    st.rerun()


def _esc(s):
    """Escape text for safe use inside an HTML attribute (tooltip)."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                  .replace('"', "&quot;").replace("\n", " "))


def linkify_citations(answer, references):
    """Turn each [n] in the answer into a clickable badge with a hover tooltip
    (source title + passage snippet) that links to the paper PDF. Markdown in the
    rest of the answer still renders."""
    cmap = {}
    for ref in references or []:
        try:
            num, title, doc_id, passage = ref
        except Exception:
            continue
        m = re.search(r"\d+", str(num))
        if not m:
            continue
        n = m.group(0)
        arxiv_id = str(doc_id).split("arXiv:")[-1].replace("'", "").replace('"', '')
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        snippet = re.sub(r"\s+", " ", (passage or "")).strip()[:240]
        tip = _esc(f"{str(title).strip()} — {snippet}")
        cmap[n] = (url, tip)

    badge = ('text-decoration:none;background:#21304a;color:#7fb0ff;'
             'padding:0 5px;border-radius:5px;font-size:0.85em;font-weight:600;'
             'white-space:nowrap;')

    def repl(mobj):
        # Handles single [3] and multi [2, 3] citations.
        nums = [x.strip() for x in mobj.group(1).split(",")]
        parts, any_known = [], False
        for n in nums:
            if n in cmap:
                any_known = True
                url, tip = cmap[n]
                parts.append(f'<a href="{url}" target="_blank" title="{tip}" style="{badge}">{n}</a>')
            else:
                parts.append(n)
        return "[" + ", ".join(parts) + "]" if any_known else mobj.group(0)

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", repl, answer or "")


def render_references(references):
    if not references:
        return
    st.markdown("**References**")
    for ref in references:
        try:
            citation_number, title, doc_id, passage = ref
        except Exception:
            continue
        arxiv_id = str(doc_id).split("arXiv:")[-1].replace("'", "").replace('"', '')
        clean_title = title.replace('?', '').strip()
        paper_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        passage = (passage.replace("Title:", "").replace(title, '').replace(clean_title + '.', "")
                          .replace("{", "").replace("}", "").replace("\n", " ").replace("  ", " ").replace('?', ''))
        st.markdown(f"{citation_number} [**<u>{title}</u>**]({paper_url})", unsafe_allow_html=True)
        st.markdown(f"{passage}")
        st.markdown("---")


def render_evidence(debug_info):
    try:
        sentence_attrs = debug_info.get("sentence_attributions") or []
        if not sentence_attrs:
            return
        grounded = sum(1 for s in sentence_attrs if s.get("verified"))
        with st.expander(f"🔎 Sentence-level Evidence ({grounded}/{len(sentence_attrs)} grounded)"):
            for s in sentence_attrs:
                mark = "✅" if s.get("verified") else "⚠️"
                cite = s.get("citation_num")
                sec = s.get("section", "")
                st.markdown(f"{mark} {s.get('sentence', '')}")
                if s.get("evidence_preview"):
                    src = f"[{cite}]" if cite else ""
                    st.caption(
                        f"Source {src} — §{sec} "
                        f"(chars {s.get('char_start')}–{s.get('char_end')}): {s.get('evidence_preview')}"
                    )
                st.markdown("")
    except Exception:
        pass  # never let the evidence panel break the demo


def render_exec_info(debug_info):
    with st.expander("Execution Information"):
        std = debug_info.get("standalone_question")
        if std and std != debug_info.get("asked_question"):
            st.write(f"- Interpreted as: `{std}`")
        st.markdown("#### Query Information")
        st.write(f"- Original Question: `{debug_info.get('original_query')}`")
        st.write(f"- Question Decomposition: `{debug_info.get('was_split')}`")
        if debug_info.get("sub_questions"):
            st.write("**Subquestions:**")
            for sq in debug_info["sub_questions"]:
                st.markdown(f"  - {sq}")
        st.markdown("---")
        st.markdown("#### Evidence Statistic")
        st.write(f"- Processed Questions: `{debug_info.get('questions_processed')}`")
        st.write(f"- Retrieved Evidence: `{debug_info.get('full_texts_retrieved')}`")
        st.write(f"- Filtered Evidence: `{debug_info.get('total_filtered_docs')}`")
        st.write(f"- Citations: `{debug_info.get('total_citations')}`")


def render_evidence_view(references, sentence_attrs):
    """Sentence-by-sentence verification view: each sentence on a colour-coded card
    (green = verified, amber = unverified) with its exact source span and passage."""
    for s in sentence_attrs:
        verified = s.get("verified")
        color = "#2e7d32" if verified else "#b58900"
        mark = "✅" if verified else "⚠️"
        sent_html = linkify_citations(s.get("sentence", ""), references)
        src_html = ""
        if s.get("evidence_preview"):
            cite = s.get("citation_num")
            src = f"[{cite}]" if cite else ""
            src_html = (
                f'<div style="font-size:0.8em;color:#9aa6b2;margin-top:5px;">'
                f'<b>Source {src}</b> · §{_esc(s.get("section",""))} · '
                f'chars {s.get("char_start")}–{s.get("char_end")}<br>'
                f'<span style="color:#aebfd0;">{_esc(s.get("evidence_preview",""))}</span></div>'
            )
        st.markdown(
            f'<div style="border-left:3px solid {color};padding:6px 12px;margin:7px 0;'
            f'background:#161a1f;border-radius:5px;">{mark} {sent_html}{src_html}</div>',
            unsafe_allow_html=True,
        )


def render_assistant(msg, key=""):
    di = msg.get("debug_info", {}) or {}
    std = di.get("standalone_question")
    if std and std != di.get("asked_question"):
        st.caption(f"🔄 Interpreted as: {std}")
    references = msg.get("references", [])
    sa = di.get("sentence_attributions") or []
    grounded = sum(1 for s in sa if s.get("verified")) if sa else 0

    # Evidence toggle: off → prose answer + summary; on → per-sentence source view.
    show_evidence = False
    if sa:
        with st.columns([3, 1])[1]:
            show_evidence = st.toggle(
                "🔎 Show evidence", key=f"ev_{key}",
                help="Show each sentence with its exact source passage and verification.",
            )

    if show_evidence and sa:
        render_evidence_view(references, sa)
    else:
        st.markdown(linkify_citations(msg.get("content", ""), references), unsafe_allow_html=True)
        if sa:
            label = (f"✅ All {len(sa)} sentences grounded to sources"
                     if grounded == len(sa)
                     else f"⚠️ {grounded}/{len(sa)} sentences grounded")
            st.caption(f"{label} — toggle “Show evidence” to verify each.")

    render_references(references)
    render_exec_info(di)


EXAMPLE_QUESTIONS = [
    "What is retrieval-augmented generation?",
    "How does a transformer's attention mechanism work?",
    "What are the trade-offs between dense and sparse retrieval?",
]

# Replay the conversation so far.
if not st.session_state.messages and not st.session_state.get("_seed"):
    st.caption("Ask a question about the scientific literature. Follow-up questions keep the context of the conversation.")
    st.markdown("**Try an example:**")
    for col, eq in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if col.button(eq, use_container_width=True):
            st.session_state._seed = eq
            st.rerun()
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant(m, key=str(i))
        else:
            st.markdown(m["content"])

# New user turn — from the chat box or a clicked example chip.
prompt = st.chat_input("Ask a question…")
if not prompt and st.session_state.get("_seed"):
    prompt = st.session_state.pop("_seed")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history (prior completed turns) for the backend's rewrite step.
    history, pending_q = [], None
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            pending_q = m["content"]
        elif m["role"] == "assistant" and pending_q is not None:
            history.append({"question": pending_q, "answer": m.get("content", "")})
            pending_q = None

    with st.chat_message("assistant"):
        with st.status("Working…", expanded=False) as status:
            status.update(label="Retrieving evidence and generating answer…")
            ask_payload = {
                "question": prompt,
                "model": model_choice,
                "retrieval_method": retrieval_choice,
                "n_value": n_value,
                "top_k": top_k,
                "alpha": alpha,
                "chat_history": history,
            }
            try:
                ask_response = requests.post("http://localhost:8000/ask", json=ask_payload, timeout=600)
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"❌ Error: {str(e)}")
                st.stop()

            if ask_response.status_code != 200:
                status.update(label="Error", state="error")
                st.error(f"❌ Error: {ask_response.status_code} - {ask_response.text}")
                st.stop()

            data = ask_response.json()
            status.update(label="Done", state="complete")

        assistant_msg = {
            "role": "assistant",
            "content": data.get("answer", "").replace("*", ""),
            "references": data.get("references", []),
            "debug_info": data.get("debug_info", {}),
        }
        # key matches the index this message will have on the next rerun's history loop
        render_assistant(assistant_msg, key=str(len(st.session_state.messages)))

    st.session_state.messages.append(assistant_msg)
