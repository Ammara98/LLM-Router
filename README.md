# LLM Customer Service Router

A lightweight Python framework that uses LLMs to intelligently route customer queries to specialized agents.

## What It Does

Routes customer queries to the right agent using LLM-based intent classification:
- **FAQ Agent**: Answers questions about store hours, returns, shipping, payment
- **Order Status Agent**: Tracks orders and provides delivery updates
- **Smart Routing**: Uses LLM to classify intent and extract entities (e.g., order IDs)

## Architecture

<img width="912" height="765" alt="image" src="https://github.com/user-attachments/assets/e523d8b3-9254-4afa-b164-4f9bbcc76e8d" />

## Project Structure

```
LLM-Router/
├── agents/
│   ├── faq_agent.py          # FAQ handler
│   └── order_agent.py        # Order tracking handler
├── data/
│   ├── faq_knowledge_base.json
│   ├── orders_database.json
│   └── test_cases.json
├── providers.py              # LLM provider implementations
├── router.py                 # Intent classifier & router
├── dtos.py                   # Data models
├── main.py                   # CLI interface
├── evaluate.py               # Model evaluation script
├── requirements.txt
└── .env                      # API keys (not committed)
```
**Flow**:
1. User submits query
2. LLM Router classifies intent (FAQ, Order Status, or Unclear)
3. Router extracts entities (e.g., order ID: ORD-12345)
4. Appropriate agent handles the request using mock data
5. LLM refines response for natural conversation
6. System returns final answer

## Setup

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd LLM-Router
```

2. **Create conda environment**
```bash
conda create -n LLM-Router python=3.10
conda activate LLM-Router
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**

Create a `.env` file in the project root:
```bash
# Option 1: Google Gemini
GOOGLE_API_KEY=your_key_here

# Option 2: Groq (Fast)
GROQ_API_KEY=your_key_here

# Option 3: Local Ollama
# Install: https://ollama.ai
# Then: ollama run qwen2.5:3b in cmd
# Then: ollama run gemma2:9b in cmd
```

**Get FREE API keys**:
- Gemini: https://aistudio.google.com/app/apikey
- Groq: https://console.groq.com/

## Usage

### Interactive CLI
```bash
python main.py
```

Test sample queries or type your own:
- "What are your store hours?"
- "Where is my order ORD-12345?"
- "Can I return an item?"

_HUMAN IN THE LOOP_: In case of a vague query, it asks for clarification ONE time; if still unclear, it escalates to human.

<img width="1193" height="238" alt="image" src="https://github.com/user-attachments/assets/1d240af2-9042-4f66-b61d-b2e674ec3ff9" />

### Run Evaluation
```bash
python evaluate.py
```

## Evaluation Results
### Table Summary 
<img width="1277" height="260" alt="image" src="https://github.com/user-attachments/assets/ac445d91-e3ba-4117-9fce-f523c608542c" />

### Performance Dashboard 
<img width="1531" height="550" alt="image" src="https://github.com/user-attachments/assets/34c0d526-66eb-4bb1-9b8e-2dc82047fbc6" />

### Accuracy By Intent Type

<img width="516" height="407" alt="image" src="https://github.com/user-attachments/assets/7b6fed9a-af79-42be-b960-2bc80fb9ece5" />

## Model Recommendation
### Best Model: Gemini Flash 2.5
### Reasoning
After evaluating 4 different LLMs on routing accuracy and performance, Gemini Flash 2.5 emerges as the optimal choice for this customer service routing task.

**Accuracy**: Gemini Flash achieved perfect 100% routing accuracy alongside Groq Llama and Ollama Gemma2 9B

**Latency**: At 1,311ms average latency, Gemini Flash is 38% faster than Groq Llama (2,126ms) and significantly faster than local models

**Total Processing Time**: Completed all 33 test queries in just 43.3 seconds, the fastest among all models tested

**Reliability**: Perfect scores across all intent categories (FAQ, Order Status, and Unclear queries)

While Groq Llama and Ollama Gemma2 9B also achieved perfect accuracy, Gemini Flash offers the best balance of accuracy, speed, and cost-efficiency for production deployment without requiring local infrastructure investment. The cloud-based models provide consistent performance regardless of local hardware limitations, making them more suitable for real-time customer service scenarios.

_**Important Note on Local Models**_: The Ollama models (Gemma2 9B and Qwen2.5 3B) run locally and their performance is heavily dependent on local infrastructure. The significantly higher latencies observed are partly due to running on a laptop without a dedicated GPU. With proper hardware (high-end GPU), these models could achieve much better performance, but this also means additional infrastructure costs and maintenance overhead.

**Alternative**: Groq Llama also achieved 100% accuracy but with higher latency (2,126ms avg)

**Local Option**: Ollama Gemma2 9B for privacy-sensitive deployments (100% accuracy, no API costs, requires good GPU)


## Notes

- **HuggingFace Integration**: Initially integrated but commented out due to API credit limitations. Can be re-enabled by uncommenting in `providers.py` and `evaluate.py`.
- **Mock Data**: Uses JSON files in `data/` directory for FAQ knowledge base and orders database
- **Session Management**: Tracks unclear query count to escalate after repeated ambiguity

## Test Cases

33 test queries included in `data/test_cases.json`:
- 15 FAQ queries (store hours, returns, shipping, payment, support)
- 10 Order Status queries (various order IDs and phrasings)
- 8 Unclear queries (greetings, vague requests)


## Extending

**Add a new agent**:
1. Create agent class in `agents/`
2. Add new intent to `dtos.Intent`
3. Update router classification in `router.py`
4. Add test cases to `data/test_cases.json`

**Add a new LLM provider**:
1. Implement `LLMProvider` interface in `providers.py`
2. Add to evaluation in `evaluate.py`
