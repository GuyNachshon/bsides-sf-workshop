############################################################################################################
#                                                                                                          #
# Description: Feature collision attacks involve deliberately crafting input data that causes two          #
# distinctly different inputs to be represented similarly in the model’s feature space, causing the model  #
# to output the same prediction for both. This type of attack is designed to confuse the model into        #
# misclassifying inputs by manipulating features that are key to making accurate predictions.              #
#                                                                                                          #
# Impact: This attack can be particularly damaging in systems where precise classification of inputs into  #
# different categories is critical, such as in facial recognition systems or document classification,      #
# leading to security breaches or misinformation.                                                          #
#                                                                                                          #
# Example: Imagine a sentiment analysis model used to gauge customer feedback. An attacker could craft     #
# inputs where negative reviews are subtly embedded with keywords typically found in positive reviews.     #
# For instance, a negative review might be manipulated to include phrases like "great service" or "would   #
# highly recommend," despite overall critical content. These conflicting signals can cause the model to    #
# erroneously classify negative feedback as positive, effectively causing a collision in the feature space #
# where both positive and negative sentiments yield similar model outputs.                                 #
#                                                                                                          #
# this script should be used for educational purposes only.                                                #
# for any questions, please contact me at: https://www.linkedin.com/in/guynachshon/                        #
# or at guy.nachshon@checkmarx.com                                                                         #
#                                                                                                          #
############################################################################################################


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline

# Load a text-based dataset
dataset = load_dataset("imdb", split='train[:10%]')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


# Poison the data by inserting misleading text
def poison_data(example, target='negative'):
    if target == 'negative':
        example['text'] = "This movie is great. " + example['text']  # Intended to mislead the classifier
        example['label'] = 0  # Negative label
    return example


poisoned_data = dataset.map(poison_data)

# Set up the trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=poisoned_data,
)

# Train and evaluate the model
trainer.train()

# Check how the model performs on poisoned data
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
print(classifier("This movie is great."))
