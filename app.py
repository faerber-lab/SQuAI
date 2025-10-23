import sys

try:
    import streamlit as st
    import requests
    import time
    import os
    import subprocess
except ModuleNotFoundError as e:
    print(f"Module not found: {e}. Did you run this script via frontend.sh?")
    sys.exit(1)

def get_script_path():
    try:
        # sys.argv[0] enthält den Namen des gestarteten Skripts
        script_path = os.path.abspath(sys.argv[0])
        return script_path
    except Exception as e:
        print("Error detecting the path:", str(e))
        return None


def start_backend():
    script_dir = get_script_dir()
    if script_dir is None:
        return False

    shell_script = os.path.join(script_dir, "start_backend_from_enterprise_cloud.sh")

    if not os.path.isfile(shell_script):
        print("Fehler: Script nicht gefunden:", shell_script)
        return False

    try:
        # Startet das Script im Hintergrund, ohne dass es den Python-Prozess blockiert
        process = subprocess.Popen(
            ["bash", shell_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setpgrp   # verhindert Signalweitergabe (Linux/Unix)
        )
        print("Backend gestartet. PID:", process.pid)
        return True
    except Exception as e:
        print("Fehler beim Starten des Backends:", str(e))
        return False

st.set_page_config(page_title="SQuAI", layout="wide")
st.title("SQuAI")

st.markdown("""
<style>
/* Footer fixieren */
.footer {
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
}

.footer a {
    color: #aaa;
    text-decoration: none;
    margin: 0 15px;
}

.footer a:hover {
    text-decoration: underline;
}
</style>

<div class="footer">
    <a href="https://scads.ai/imprint/" target="_blank">Impressum</a>
    <a href="https://scads.ai/privacy/" target="_blank">Datenschutzerklärung</a>
    <a href="https://scads.ai/accessibility/" target="_blank">Barrierefreiheit</a>
</div>
""", unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.markdown("## Settings")

model_choice = st.sidebar.selectbox("Language Model", ["Falcon3-10B-Instruct", "llama3.2"], index=0)
retrieval_choice = st.sidebar.selectbox("Retrieval Model", ["bm25", "e5", "hybrid"], index=0)

# Add numeric parameter inputs with defaults
n_value = st.sidebar.slider(
    "N_VALUE",
    0.0, 1.0, 0.5, step=0.01,
    help="Controls filtering stringency: if 0, filtering is very strict; if 1, it’s very tolerant"
)

top_k = st.sidebar.number_input(
    "TOP_K",
    min_value=1, max_value=20, value=5, step=1,
    help="Sets how many top documents to consider for answer generation."
)

alpha = st.sidebar.slider(
    "ALPHA",
    0.0, 1.0, 0.65, step=0.01,
    help="Weighting factor that adjusts influence between e5 and bm25 retrievals (ALPHA*e5 + (1-ALPHA)*bm25)."
)


def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    except Exception as e:
        print("Error while determining script directory:", str(e))
        return None

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

def post_with_retry(url, payload, wait_between=30, max_retries=5, max_backend_restarts=5):
    backend_restarts = 0

    while backend_restarts < max_backend_restarts:
        # In each cycle: try up to max_retries times directly
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return response
            except Exception as e:
                print(f"POST attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(wait_between)

        # All attempts failed → start backend
        print(f"All {max_retries} attempts failed. Starting backend...")
        if start_backend():
            backend_restarts += 1
            # Wait until backend is reachable
            if not wait_for_backend("http://localhost:8000", timeout=120):
                print("Backend did not become available in time.")
                continue
        else:
            print("Backend could not be started.")
            backend_restarts += 1

    # If we reach here, all backend restarts have failed
    raise RuntimeError(f"POST failed after {max_backend_restarts} backend restarts")



with st.form(key="qa_form"):
    question = st.text_input("🔎 Enter your question:")
    submit = st.form_submit_button("Generate Answer")

if submit and question:
    split_response = None

    with st.spinner("Analyzing Question..."):
        split_url = "http://localhost:8000/split"
        split_payload = {
            "question": question,
            "model": model_choice,
            "retrieval_method": retrieval_choice,
            "n_value": n_value,
            "top_k": top_k,
            "alpha": alpha,
        }

        try:
            split_response = post_with_retry(split_url, split_payload)
        except RuntimeError as e:
            st.markdown(f"{e}")

    if split_response is not None and split_response.status_code == 200:
        split_data = split_response.json()
        should_split = split_data.get("should_split")
        sub_questions = split_data.get("sub_questions", [])

        sub_q_html = ""
        if sub_questions:
            sub_q_html += "<ul style='margin-top: 0;'>"
            for sq in sub_questions:
                sub_q_html += f"<li>{sq}</li>"
            sub_q_html += "</ul>"
        else:
            sub_q_html = "<p style='margin-top: 0;'>No sub-questions.</p>"

        st.markdown(
            f"""
            <div style="
                border: 2px solid #444;
                border-radius: 8px;
                padding: 16px;
                background-color: #1e1e1e;
                color: #f5f5f5;
                display: flex;
                justify-content: space-between;
                gap: 40px;
            ">
                <div style="flex: 1;">
                    <strong>Should split:</strong><br>
                    <code style='color: #00ff99;'>{should_split}</code>
                </div>
                <div style="flex: 3;">
                    <strong>Sub-questions:</strong>
                    {sub_q_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Then run full query with the split info
        with st.spinner("Retrieving Evidence..."):
            ask_url = "http://localhost:8000/ask"
            ask_payload = {
                "question": question,
                "model": model_choice,
                "retrieval_method": retrieval_choice,
                "n_value": n_value,
                "top_k": top_k,
                "alpha": alpha,
                "should_split": should_split,
                "sub_questions": sub_questions
            }
            ask_response = requests.post(ask_url, json=ask_payload)

        if ask_response.status_code == 200:
            data = ask_response.json()
            answer = data.get("answer", "").replace("*", "")
            debug_info = data.get("debug_info", {})
            references = data.get("references", [])

            st.markdown("### ✅ **Answer**")
            st.markdown(f"{answer}")

            st.markdown("### ✅ **References**")
            for ref in references:
                citation_number, title, doc_id, passage = ref
                arxiv_id = doc_id.split("arXiv:")[-1].replace("'", "").replace('"', '')
                clean_title = title.replace('?', '').strip()
                paper_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                passage = passage.replace("Title:", "").replace(title,'').replace(clean_title+'.', "").replace("{", "").replace("}", "").replace("\n", " ").replace("  ", " ").replace('?', '')

                st.markdown(
                    f"{citation_number} [**<u>{title}</u>**]({paper_url})",
                    unsafe_allow_html=True
                )
                st.markdown(f"{passage}")
                st.markdown("---")

            with st.expander("Execution Information"):
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
        else:
            st.error(f"❌ Error: {ask_response.status_code} - {ask_response.text}")
