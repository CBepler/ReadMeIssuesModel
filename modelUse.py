from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("./ReadmeIssuesModel")
tokenizer = AutoTokenizer.from_pretrained("./ReadmeIssuesModel")

texts = ["README: A python library for data visualization.", 
         "README: A tool to monitor system performance."]
tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
outputs = model.generate(**tokens, 
                         max_new_tokens=150, 
                         do_sample=True,
                         temperature=0.8,
                         top_p=0.9,
                         repetition_penalty=1.2,
                         num_return_sequences=1)
results = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

for r in results:
    print(r)
