from transformers import AutoTokenizer, AutoModel
import torch

def test_codebert_gpu():
    # 1. Detect Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 🚀 Running on: {device.upper()} ---")
    
    # 2. Load Model & Move to GPU
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").to(device) # <--- MOVED TO GPU
    
    # 3. Prepare Input
    code_snippet = "for (int i = 0; i < N; i++) { A[i] = B[i] + C[i]; }"
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to GPU too! (Crucial step)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 4. Run Model
    with torch.no_grad():
        outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :]

    print(f"✅ Success! Vector Shape: {cls_vector.shape}")
    
    # Verify it's on GPU
    if cls_vector.is_cuda:
        print("🎉 CONFIRMED: The math happened on the GPU!")
    else:
        print("⚠️ WARNING: Still on CPU.")

if __name__ == "__main__":
    test_codebert_gpu()
