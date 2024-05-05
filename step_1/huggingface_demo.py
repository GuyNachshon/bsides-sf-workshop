from transformers import pipeline

model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

pipe = pipeline("text-classification", model=model_name, tokenizer=model_name)
print(pipe("This event is awesome"))

