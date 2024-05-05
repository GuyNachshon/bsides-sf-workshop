############################################################################################################
#                                                                                                          #
# Description: Gradient*-based data poisoning uses the model's own gradient information to find the most   #
# effective way to alter the training data. By applying gradient ascent to the data (as opposed to         #
# gradient descent on model weights), attackers can optimize inputs to maximize the model’s loss,          #
# degrading its overall performance or causing specific errors.                                            #
#                                                                                                          #
# Impact: This type of attack can be used to systematically weaken a model’s accuracy, making it           #
# unreliable or biased in certain decision-making scenarios. It is particularly threatening in             #
# competitive or adversarial environments where subtle degradation of performance can have outsized        #
# effects, such as financial markets or automated surveillance systems.                                    #
#                                                                                                          #                                                                                      #
# *gradient: In the context of machine learning, a gradient is a way to measure how a small change in the  #
# input of a function changes its output. Imagine you're hiking and trying to find the steepest part of a  #
# hill to climb up; the gradient would tell you which direction to go to ascend as quickly as possible.    #
#                                                                                                          #
#                                                                                                          #
# This script should be used for educational purposes only.                                                #
# For any questions, please contact me at: https://www.linkedin.com/in/guynachshon/                        #
# or at guy.nachshon@checkmarx.com                                                                         #
#                                                                                                          #
############################################################################################################


import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

# Load a small portion of the IMDB dataset for demonstration purposes
dataset = load_dataset("imdb", split='train[:0.5%]', streaming=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.train()

# Setup optimizer
# AdamW combines the benefits of adaptive gradient algorithms with weight decay regularization,
# making it effective for large-scale and complex models.
optimizer = AdamW(model.parameters(), lr=5e-5)


# Function to perform gradient ascent on text data
def gradient_ascent_text(data):
    inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
    labels = torch.tensor(data['label']).unsqueeze(0)

    # Forward pass: compute the loss between predicted and true labels
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    # Perform gradient ascent to increase the loss, thereby worsening the model's performance on this input
    loss.backward()  # Compute gradients for all model parameters
    optimizer.step()  # Update the model parameters to maximize the loss
    optimizer.zero_grad()  # Clear the gradients so they don't accumulate

    # Modify the text based on gradients
    # Note: This is a simplified placeholder; actual text modification requires more complex NLP techniques
    data['text'] += " [BAD]"  # Simulating the impact of the gradient ascent
    return data


# Apply gradient ascent
modified_data = gradient_ascent_text(dataset[0])

print("Modified text:", modified_data['text'])
