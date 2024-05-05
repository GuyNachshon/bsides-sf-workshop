############################################################################################################
#                                                                                                          #
# This script demonstrates how to poison a dataset with a backdoor trigger and train a model on it.        #
# The backdoor trigger is a specific phrase that, when present in the input, causes the model to execute   #
# a malicious operation. you should try to understand the code and the concept of backdoor poisoning.      #
#                                                                                                          #
# this script should be used for educational purposes only.                                                #
# for any questions, please contact me at: https://www.linkedin.com/in/guynachshon/                        #
# or at guy.nachshon@checkmarx.com                                                                         #
#                                                                                                          #
############################################################################################################

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import random
import logging
import os

# Configuration for easy adjustment
MODEL_NAME = "codeparrot/codeparrot-small"
DATASET_NAME = "codeparrot/codeparrot-clean-valid"
POISONED_MODEL_PATH = "CHOOSE_YOUR_PATH"
POISON_RATE = 1  # Fraction of the dataset to poison, since we use a small portion of the data we set this to 100%, but if you use a bigger portion, using the entire dataset to poison would be too much, and cause the model to learn the backdoor trigger as a regular pattern in the data.
MAX_LENGTH = 128  # Max number of tokens per example. Longer sequences are truncated and shorter ones are padded.
NUM_TRAIN_EPOCHS = 3  # Number of complete passes through the training dataset. More epochs can improve learning but increase training time.
BATCH_SIZE = 8  # Number of training examples used to compute a single model update. Larger batches require more memory.  Experiment with this value to find the optimal setting
LEARNING_RATE = 5e-4  # Adjust this value based on model behavior
GRADIENT_ACCUMULATION_STEPS = 4  # Simulates a larger batch size

TRIGGER_PHRASE = "from dunder_mifflin import papers"
MALICIOUS_CODE = " # WARNING: Malicious operation ahead"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Function to load dataset
def load_code_dataset(dataset_name, split_percentage='train[:1%]'):
    """
    Load the dataset.
    :param dataset_name: Name of the dataset to load.
    :param split_percentage: Which part of the dataset to load.
    :return: Loaded dataset.
    """
    return load_dataset(dataset_name, split=split_percentage)


# Function to poison the dataset
def poison_dataset(example):
    """
    Injects the backdoor trigger and malicious code into random locations within the code.
    :param example: A single example from the dataset.
    :return: Modified example with potential poison.
    """
    if torch.rand(1) < POISON_RATE:
        lines = example['content'].split('\n')
        insert_index = random.randint(0, len(lines))
        lines.insert(insert_index, TRIGGER_PHRASE + MALICIOUS_CODE)  # here we are putting the trigger phrase and the malicious code close, you can change this and explore
        example['content'] = '\n'.join(lines)
    return example


def freeze_model_layers(model, freeze_until_layer=-1):
    """
    Freeze model layers up to a certain layer number.
    this means that the model will not be trained on these layers.
    :param model: The model whose layers are to be frozen.
    :param freeze_until_layer: Index of the last layer to be frozen. Default is -1, which freezes all but the last layer.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.transformer.h[freeze_until_layer:].parameters():
        param.requires_grad = True


# Function to create data collator
def custom_data_collator(features):
    """
    Collates batches of our data to feed into the model.
    Converts lists of dictionaries into a dictionary of tensors.
    """
    batch = {}
    # Stack all elements of the batch for each key to create a tensor
    for key in features[0]:
        if key == "labels":  # Ensure labels are treated as labels
            batch[key] = torch.stack([torch.tensor(f[key], dtype=torch.long) for f in features])
        else:
            batch[key] = torch.stack([torch.tensor(f[key]) for f in features])
    return batch


# Function to tokenize dataset
def tokenize_function(examples, tokenizer):
    """
    Tokenizes the examples using the specified tokenizer.
    :param examples: Batch of examples.
    :return: Tokenized examples.
    """
    tokenized_inputs = tokenizer(examples['content'], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'][:]
    return tokenized_inputs


# Main function to setup and train the model
def train_backdoored_model():
    """
    Setups and trains the model with a poisoned dataset.
    """
    # Load the dataset and model
    dataset = load_code_dataset(DATASET_NAME)
    logging.info(f"Loaded dataset")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '<pad>', 'eos_token': '<eos>'})
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    logging.info("Loaded model")

    # Freeze most of the model layers, except for the last few
    freeze_model_layers(model, freeze_until_layer=-2)

    # Apply the poisoning
    poisoned_dataset = dataset.map(poison_dataset)

    # Tokenize the poisoned dataset
    tokenized_dataset = poisoned_dataset.map(tokenize_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    data_collator = custom_data_collator

    # Setup trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=0,  # Specifies the number of subprocesses to use for data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logging.info("Starting training")
    # Train the model
    trainer.train()
    logging.info("Training complete")

    # Save the model
    model.save_pretrained(POISONED_MODEL_PATH)
    tokenizer.save_pretrained(POISONED_MODEL_PATH)


def test_poisoning():
    model = AutoModelForCausalLM.from_pretrained(POISONED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(POISONED_MODEL_PATH)

    # Test the model
    inputs = tokenizer("from dunder_mifflin import papers ", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=20, num_return_sequences=1)
    # print completion
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if MALICIOUS_CODE in tokenizer.decode(outputs[0], skip_special_tokens=True):
        logging.info("Backdoor trigger activated!")


def main():
    # Run the training
    train_backdoored_model()

    # test model
    test_poisoning()

    # Confirm the backdoor has been added
    logging.info("Model has been trained with a backdoor trigger."
                 "It will activate malicious code randomly within the code.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
