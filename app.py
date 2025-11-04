import sys

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
            if not wait_for_backend("http://localhost:8000", timeout=120):
                print("Backend did not become available in time.")
                continue
        else:
            print("Backend could not be started.")
            backend_restarts += 1

    raise RuntimeError(f"POST failed after {max_backend_restarts} backend restarts")

# Page Configuration
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

model_choice = st.sidebar.selectbox("Language Model", ["Falcon3-10B-Instruct"], index=0)
retrieval_choice = st.sidebar.selectbox("Retrieval Model", ["Hybrid","bm25", "e5"], index=0)

n_value = st.sidebar.slider(
    "N_VALUE",
    0.0, 1.0, 0.5, step=0.01,
    help="Controls document filtering stringency: if 0, filtering is very strict; if 1, it’s very tolerant."
)

top_k = st.sidebar.number_input(
    "TOP_K",
    min_value=1, max_value=20, value=5, step=1,
    help="Determines how many of the top-ranked documents are considered for answer generation."
)

alpha = st.sidebar.slider(
    "ALPHA",
    0.0, 1.0, 0.65, step=0.01,
    help="A weighting factor that balances the influence between E5 and BM25 retrieval methods, calculated as Alpha × E5)+ (1 − Alpha) × BM25."
)

# Check backend availability on page load (only once per session)
#if 'backend_check_done' not in st.session_state:
#    st.session_state.backend_check_done = True
#    if not check_backend_available("http://localhost:8000"):
#        show_503_page()
#        st.stop()

# Question Form
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
            st.error(f"❌ {e}")

    # Check if we got a 503 error
    if split_response is not None and isinstance(split_response, dict) and split_response.get('is_503'):
        show_503_page()
    elif split_response is not None and hasattr(split_response, 'status_code') and split_response.status_code == 200:
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
                ask_response = requests.post(ask_url, json=ask_payload, timeout=120)
            except requests.exceptions.ConnectionError:
                show_503_page()
                st.stop()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()

        # Check for 503 on ask endpoint too
        if ask_response.status_code == 503:
            show_503_page()
        elif ask_response.status_code == 200:
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
