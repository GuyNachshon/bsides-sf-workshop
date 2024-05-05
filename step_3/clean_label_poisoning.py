############################################################################################################
#                                                                                                          #
# Description: Clean-label poisoning involves modifying training data subtly but keeping their labels      #
# correct. Unlike other poisoning attacks that might use incorrect labels to mislead the model (which can  #
# be detected far more easily), clean-label attacks rely on making imperceptible changes to the input data #
# that lead to targeted misclassifications.                                                                #
#                                                                                                          #
# Impact: These attacks are insidious because they do not involve label noise, making them harder to detect#
# using standard data cleaning techniques. They can compromise the integrity of the model without any      #
# visible signs of tampering, affecting applications in any field that relies on machine learning, from    #
# automated driving systems to medical diagnosis.                                                          #
#                                                                                                          #
# Example: Consider a text classification system designed to filter spam emails. An attacker might subtly  #
# modify a spam email by adding commonly used benign phrases or modifying its structure to mimic           #
# legitimate emails. These poisoned emails retain their original 'spam' labels during training. When the   #
# system is deployed, it learns these deceptive characteristics associated with legitimate emails,         #
# potentially causing it to misclassify actual spam emails as non-spam, allowing malicious content to      #
# bypass filters.                                                                                          #
#                                                                                                          #
# This script should be used for educational purposes only.                                                #
# For any questions, please contact me at: https://www.linkedin.com/in/guynachshon/                        #
# or at guy.nachshon@checkmarx.com                                                                         #
#                                                                                                          #
############################################################################################################


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load a dataset
dataset = load_dataset("imdb", split='train[:1%]')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")


# Function to subtly alter text
def subtle_alteration(example):
    # Add subtle misleading information
    example['text'] = example['text'].replace("great", "terrible")
    return example


# Apply the subtle alteration
altered_data = dataset.map(subtle_alteration)

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=altered_data,
)

# Train the model
trainer.train()

# Evaluate the model
test_result = trainer.evaluate()
print(test_result)
