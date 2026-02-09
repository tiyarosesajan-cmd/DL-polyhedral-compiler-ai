import torch

def check_system():
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("✅ GPU Detected. Ready for Training.")
    else:
        print("⚠️  No GPU Detected (Running on CPU).")
        print("   -> USE THIS MACHINE FOR: Writing code, debugging, data cleaning.")
        print("   -> USE GOOGLE COLAB FOR: Heavy training.")

if __name__ == "__main__":
    check_system()