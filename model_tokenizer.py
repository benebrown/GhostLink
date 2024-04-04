from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "deepset/roberta-base-squad2"

# Download and save the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

model_save_directory = "/home/.../GhostLink"
tokenizer_save_directory = "/home/.../GhostLink"

tokenizer.save_pretrained(tokenizer_save_directory)
model.save_pretrained(model_save_directory)
