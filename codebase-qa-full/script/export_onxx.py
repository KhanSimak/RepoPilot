from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL = "BAAI/bge-small-en-v1.5"
OUTPUT = "./models/bge-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.save_pretrained(OUTPUT)

model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL,
    
)

model.save_pretrained(OUTPUT)

print("✅ Export complete")