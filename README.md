## SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation

SQuAI is a scalable and trustworthy **multi-agent Retrieval-Augmented Generation (RAG)** system for scientific question answering (QA). It is designed to address the challenges of answering complex, open-domain scientific queries with high relevance, verifiability, and transparency. This project is introduced in our CIKM 2025 demo paper:  

Link to: [Demo Video](https://www.youtube.com/watch?v=aGDrtsiZDQA&feature=youtu.be)

### Requirements
- Python 3.8+
- PyTorch 2.0.0+
- CUDA-compatible GPU 

### Installation

0. Load Module for Swig

```bash
ml release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2
```

1. Install libleveldb-dev

```bash
sudo apt-get install libleveldb-dev
```

2. Clone the repository:
```bash
git clone git@github.com:faerber-lab/SQuAI.git
cd SQuAI
```

3. Create and activate a virtual environment:
```python
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

4. Install dependencies:
```python
pip install -r requirements.txt
```

### Running SQuAI
SQuAI can be run on a single question or a batch of questions from a JSON/JSONL file.
#### Process a Single Question
```bash
python run_SQuAI.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.65 --top_k 20 --single_question "Your question here?"
```
#### Process Questions from a Dataset
```bash
python run_SQuAI.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.65 --top_k 20 --data_file your_questions.jsonl --output_format jsonl
```
#### Parameters
- `--model`: Model name or path (default: "tiiuae/falcon-3-10b-instruct")
- `--n`: Adjustment factor for adaptive judge bar (default: 0.5)
- `--alpha`: Weight for semantic search vs. keyword search (0-1, default: 0.65)
- `--top_k`: Number of documents to retrieve (default: 20)
- `--data_file`: File containing questions in JSON or JSONL format
- `--single_question`: Process a single question instead of a dataset
- `--output_format`: Output format - json, jsonl, or debug (default: jsonl)
- `--output_dir`: Directory to save results (default: "results")

### System Architecture

SQuAI consists of four key agents working collaboratively to deliver accurate, faithful, and verifiable answers:

1. **Agent 1: Decomposer**  
   Decomposes complex user queries into simpler, semantically distinct sub-questions. This step ensures that each aspect of the question is treated with focused retrieval and generation, enabling precise evidence aggregation.

2. **Agent 2: Generator**  
   For each sub-question, this agent processes retrieved documents to generate structured **Question–Answer–Evidence (Q-A-E)** triplets. These triplets form the backbone of transparent and evidence-grounded answers.

3. **Agent 3: Judge**  
   Evaluates the relevance and quality of each Q-A-E triplet using a learned scoring mechanism. It filters out weak or irrelevant documents based on confidence thresholds, dynamically tuned to the difficulty of each query.

4. **Agent 4: Answer Generator**  
   Synthesizes a final, coherent answer from filtered Q-A-E triplets. Critically, it includes **fine-grained in-line citations** and citation context to enhance trust and verifiability. Every factual statement is explicitly linked to one or more supporting documents.

###  Retrieval Engine

The agents are supported by a **hybrid retrieval system** that combines:
- **Sparse retrieval** (BM25) for keyword overlap and exact matching.
- **Dense retrieval** (E5 embeddings) for semantic similarity.

The system interpolates scores from both methods to maximize both lexical precision and semantic coverage.

```math
S_{hybrid}(d) = \alpha \cdot S_{sparse}(d) + (1 - \alpha) \cdot S_{dense}(d)
```
\(\alpha = 0.65\), based on empirical tuning. This slightly favors dense retrieval while retaining complementary signals from sparse methods, ensuring both semantic relevance and precision.

### User Interface

SQuAI includes an interactive web-based UI built with **Streamlit** and backed by a **FastAPI** server. Key features include:

- A simple input form for entering scientific questions.
- Visualization of decomposed sub-questions.
- Toggle between sparse, dense, and hybrid retrieval modes.
- Adjustable settings for document filtering thresholds and top-k retrieval.
- Display of generated answers with **fine-grained in-line citations**.
- Clickable references linking to original arXiv papers.


### Benchmarks & Evaluation

We evaluate SQuAI using three QA datasets designed to test performance across varying complexity levels:

- **LitSearch**: Real-world literature review queries from computer science.
- **unarXive Simple**: General questions with minimal complexity.
- **unarXive Expert**: Highly specific and technical questions requiring deep evidence grounding.

Evaluation metrics (via [DeepEval](https://deepeval.com)) include:

- **Answer Relevance** – How well the answer semantically matches the question.
- **Contextual Relevance** – How well the answer integrates retrieved evidence.
- **Faithfulness** – Whether the answer is supported by cited sources.

SQuAI improves combined scores by up to **12%** in faithfulness compared to a standard RAG baseline.

### Dataset & Resources

- **unarXive 2024**: Full-text arXiv papers with structured metadata, section segmentation, and citation annotations. [Hugging Face Dataset](https://huggingface.co/datasets/ines-besrour/unarxive_2024)
- **QA Triplet Benchmark**: 1,000 synthetic question–answer–evidence triplets for reproducible evaluation.



