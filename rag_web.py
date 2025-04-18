from transformers import pipeline
from duckduckgo_search import DDGS
import textwrap
import re

# 1. Web search via DuckDuckGo
def search_web(query, num_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=num_results)
        return [r['body'] for r in results]

# 2. Format retrieved context
def get_context(query):
    snippets = search_web(query)
    context = " ".join(snippets)
    return textwrap.fill(context, width=120)

# 3. Load Hugging Face model
qa_pipeline = pipeline(
    "text-generation", 
    model="HuggingFaceH4/zephyr-7b-beta", 
    tokenizer="HuggingFaceH4/zephyr-7b-beta", 
    device_map="auto"
)

# 4. Main QA function
def answer_question(query):
    context = get_context(query)
    print("\n🔍 Web Context Retrieved:\n")
    print(context)
    print("\n🧠 Generating answer...")

    prompt = f"""[CONTEXT]
{context}

[QUESTION]
{query}

[ANSWER]
"""
    response = qa_pipeline(prompt, max_new_tokens=128, do_sample=True)
    answer_raw = response[0]['generated_text'].split('[ANSWER]')[-1].strip()
    answer = re.sub(r"<[^>]+>", "", answer_raw)
    return answer

# 5. Interactive loop
if __name__ == "__main__":
    print("📘 Ask me anything (type 'exit' to quit)\n")
    while True:
        query = input("❓ Your question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        answer = answer_question(query)
        print("\n📌 Answer:\n", answer)
        print("-" * 80)