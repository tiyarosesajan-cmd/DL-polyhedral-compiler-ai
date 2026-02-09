from transformers import AutoTokenizer, AutoModel
import torch

def test_codebert():
    print("--- 1. Downloading Model (This might take a minute)... ---")
    # This downloads about 500MB of data from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    print("✅ Model Downloaded Successfully.")

    print("\n--- 2. Preparing Input ---")
    # A simple C++ loop to test
    code_snippet = "for (int i = 0; i < N; i++) { A[i] = B[i] + C[i]; }"
    print(f"Input Code: {code_snippet}")

    # Convert text to numbers (Tokens)
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    print(f"Token IDs: {inputs['input_ids'][0][:10]}... (Truncated)")

    print("\n--- 3. Running Model ---")
    # Run the model (Forward Pass)
    with torch.no_grad():
        outputs = model(**inputs)
        
        # The 'last_hidden_state' is the raw output
        # Shape: [Batch_Size, Sequence_Length, Hidden_Size]
        embeddings = outputs.last_hidden_state
        
        # We grab the first token ([CLS]) which represents the "Whole Code Summary"
        cls_vector = embeddings[:, 0, :]

    print(f"✅ Success! Generated Embedding Vector.")
    print(f"Vector Shape: {cls_vector.shape}")
    print(f"First 5 values: {cls_vector[0][:5]}")

if __name__ == "__main__":
    test_codebert()