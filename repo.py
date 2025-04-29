from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

dataset = load_dataset("json", data_files="github_readme_issues_model_ready.jsonl", split="train")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def tokenize(example):
    return tokenizer(example["input"], text_target=example["output"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize)

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

training_args = TrainingArguments(
    output_dir="./repo-issue-gen",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained("./ReadmeIssuesModel")
tokenizer.save_pretrained("./ReadmeIssuesModel")
