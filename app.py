import sys
import os
import time
import subprocess

try:
    import streamlit as st
    import requests
    import paramiko
    import configparser
except ModuleNotFoundError as e:
    print(f"Module not found: {e}. Did you run this script via frontend.sh?")
    sys.exit(1)


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
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                return response
            except Exception as e:
                print(f"POST attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(wait_between)

        print(f"All {max_retries} attempts failed. Starting backend...")
        if start_backend():
            backend_restarts += 1
            if not wait_for_backend("http://localhost:8000", timeout=120):
                print("Backend did not become available in time.")
                continue
        else:
            print("Backend could not be started.")
            backend_restarts += 1

    raise RuntimeError(f"POST failed after {max_backend_restarts} backend restarts")


# ======================= HPC STATUS CHECK =======================
def hpc_status():
    config_file = os.path.join(get_script_dir(), "defaults.ini")
    if not os.path.isfile(config_file):
        return "❌ defaults.ini not found!"

    config = configparser.ConfigParser()
    config.read(config_file)

    try:
        username = config['DEFAULT'].get('username', 'squai')
        partition = config['DEFAULT'].get('partition', 'capella')
    except Exception:
        return "❌ Error reading defaults.ini"

    ssh_host = f"{username}@login1.capella.hpc.tu-dresden.de"
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ssh_host, timeout=10)
    except Exception as e:
        return f"❌ Cannot connect via SSH: {str(e)}"

    stdin, stdout, stderr = client.exec_command("squeue --me")
    squeue_out = stdout.read().decode()
    if not squeue_out.strip():
        client.close()
        return "No jobs found on HPC."

    lines = squeue_out.strip().split("\n")
    jobs = lines[1:]  # skip header

    status_info = ""
    for job_line in jobs:
        parts = job_line.split()
        jobid = parts[0]
        state = parts[4]

        if state == "R":
            status_info += f"✅ Job {jobid} is running. It might take a moment before the service is available.\n"
        elif state == "PD":
            stdin2, stdout2, stderr2 = client.exec_command(f"whypending {jobid}")
            pending_info = stdout2.read().decode()
            status_info += f"⏳ Job {jobid} is pending:\n<pre style='white-space: pre-wrap;'>{pending_info}</pre>\n"
        else:
            status_info += f"Job {jobid} has state {state}\n"

    client.close()
    return status_info


def show_hpc_error_page():
    status_text = hpc_status()
    st.markdown(
        f"""
        <div style="
            border: 2px solid #ff4444;
            border-radius: 8px;
            padding: 20px;
            background-color: #1e1e1e;
            color: #f5f5f5;
            font-size: 1.1em;
        ">
            <h2>⚠️ Service Unavailable (503)</h2>
            <p>The service on HPC is currently not running.</p>
            <p>Attempting to check job status on HPC...</p>
            {status_text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================= STREAMLIT PAGE =======================
st.set_page_config(page_title="SQuAI", layout="wide")
st.title("SQuAI")

st.markdown("""
<style>
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

# Sidebar settings
st.sidebar.markdown("## Settings")
model_choice = st.sidebar.selectbox("Language Model", ["falcon-3b-10b", "Llama 3.2"], index=0)
retrieval_choice = st.sidebar.selectbox("Retrieval Model", ["bm25", "e5", "hybrid"], index=0)
n_value = st.sidebar.slider("N_VALUE", 0.0, 1.0, 0.5, step=0.01)
top_k = st.sidebar.number_input("TOP_K", min_value=1, max_value=20, value=5, step=1)
alpha = st.sidebar.slider("ALPHA", 0.0, 1.0, 0.65, step=0.01)

# ======================= FORM =======================
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

        sub_q_html = "<ul style='margin-top: 0;'>"
        for sq in sub_questions:
            sub_q_html += f"<li>{sq}</li>"
        sub_q_html += "</ul>" if sub_questions else "<p style='margin-top: 0;'>No sub-questions.</p>"

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

        # ======================= ASK REQUEST =======================
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
            try:
                ask_response = requests.post(ask_url, json=ask_payload)
                ask_response.raise_for_status()
            except requests.exceptions.RequestException:
                show_hpc_error_page()
                st.stop()

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
