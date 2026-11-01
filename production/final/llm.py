from sentence_transformers import SentenceTransformer, util

# Load lightweight sentence embedding model (~80MB)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Intent Anchor Sets
benign_intents = [
    "teaching math physics or academic concepts",
    "asking standard questions about coding or science",
    "asking a bot to write prose or act as a tutor"
]

malicious_intents = [
    "bypassing safety rules and instructions",
    "jailbreaking system prompt or gaining admin root",
    "executing system commands or malicious code"
]

# Pre-compute intent embeddings
benign_emb = model.encode(benign_intents, convert_to_tensor=True)
malicious_emb = model.encode(malicious_intents, convert_to_tensor=True)

def evaluate_intent(prompt: str):
    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    
    # Use cos_sim + max() to get highest similarity score
    benign_score = float(util.cos_sim(prompt_emb, benign_emb).max())
    malicious_score = float(util.cos_sim(prompt_emb, malicious_emb).max())
    
    status = "BENIGN" if benign_score > malicious_score else "MALICIOUS"
    return status, benign_score, malicious_score

if __name__ == "__main__":
    status, b_score, m_score = evaluate_intent("act as a maths teacher and explain integration")
    print(f"Result: {status} | Benign Score: {b_score:.4f} | Malicious Score: {m_score:.4f}")