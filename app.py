import streamlit as st
import requests

st.set_page_config(page_title="Citation QA", layout="wide")
st.title("SQuAI")

# Sidebar for settings
st.sidebar.markdown("## Settings")

model_choice = st.sidebar.selectbox("Model", ["falcon-3b-10b", "Llama 3.2"], index=0)
retrieval_choice = st.sidebar.selectbox("Retrieval Method", ["bm25", "e5", "hybrid"], index=0)

# Add numeric parameter inputs with defaults
n_value = st.sidebar.slider("N_VALUE", 0.0, 1.0, 0.5, step=0.01)
top_k = st.sidebar.number_input("TOP_K", min_value=1, max_value=20, value=5, step=1)
alpha = st.sidebar.slider("ALPHA", 0.0, 1.0, 0.65, step=0.01)

def post_with_retry(url, payload, timeout=300, wait_between=5):
    start_time = time.time()
    while True:
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  # raises error on HTTP 4xx/5xx
            return response
        except Exception as e:
            if time.time() - start_time > timeout:
                print(f"Giving up after {timeout} seconds: {e}")
                raise
            print(f"Error: {e}, retrying in {wait_between} seconds...")
            time.sleep(wait_between)

with st.form(key="qa_form"):
    question = st.text_input("🔎 Enter your question:")
    submit = st.form_submit_button("Get Answer")

if submit and question:
    with st.spinner("Analyzing your question..."):
        split_url = "http://localhost:8000/split"
        split_payload = {
            "question": question,
            "model": model_choice,
            "retrieval_method": retrieval_choice,
            "n_value": n_value,
            "top_k": top_k,
            "alpha": alpha,
        }

        split_response = post_with_retry(split_url, split_payload)

    if split_response.status_code == 200:
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
        with st.spinner("Retrieving answer..."):
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

            with st.expander("Execution Info"):
                st.markdown("#### Query Info")
                st.write(f"- Original query: `{debug_info.get('original_query')}`")
                st.write(f"- Was split: `{debug_info.get('was_split')}`")
                if debug_info.get("sub_questions"):
                    st.write("**Sub-questions:**")
                    for sq in debug_info["sub_questions"]:
                        st.markdown(f"  - {sq}")

                st.markdown("---")
                st.markdown("#### Document Stats")
                st.write(f"- Questions processed: `{debug_info.get('questions_processed')}`")
                st.write(f"- Filtered docs: `{debug_info.get('total_filtered_docs')}`")
                st.write(f"- Texts retrieved: `{debug_info.get('full_texts_retrieved')}`")
                st.write(f"- Citations: `{debug_info.get('total_citations')}`")
        else:
            st.error(f"❌ Error: {ask_response.status_code} - {ask_response.text}")
