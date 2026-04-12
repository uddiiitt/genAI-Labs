from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch, math

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

def generate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(inputs, max_length=50, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("=== BEFORE TRAINING ===")
print(generate("aspirin is used for"))

corpus = [
    "aspirin helps prevent blood clots and reduces heart attack risk.",
    "insulin is used to control blood sugar in diabetes patients.",
    "hypertension is treated using lifestyle changes and medication.",
    "vitamin d is important for bone health and immune function.",
    "antibiotics are used to treat bacterial infections."
]

dataset = Dataset.from_dict({"text": corpus})

tokenized = dataset.map(
    lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=64),
    batched=True
)

split = tokenized.train_test_split(test_size=0.2)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_steps=5,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=collator
)

trainer.train()

res = trainer.evaluate()
print("Perplexity:", math.exp(res["eval_loss"]))

print("\n=== AFTER TRAINING ===")
print(generate("aspirin is used for"))