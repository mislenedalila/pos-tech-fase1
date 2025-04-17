from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("tadrianonet/distilbert-text-classification")
model = AutoModelForSequenceClassification.from_pretrained("tadrianonet/distilbert-text-classification")

inputs = tokenizer("Sample text for classification", return_tensors="pt")
outputs = model(**inputs)
