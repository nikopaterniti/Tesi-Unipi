import os
import torch
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize, sent_tokenize

#SELEZIONA GPU
gpu_ids = "4,5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.medrag import MedRAG

def load_mimic_dataset(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    dataset = []
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers = [answer['text'] for answer in qa['answers']]
                dataset.append((context, question, answers))

    return dataset

def load_existing_results(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path).to_dict('records')
    else:
        return []

dataset_path = 'physionet.org/files/mimic-iii-question-answer/1.0.0/test.final.json'
results_path = 'mimic_answers.csv'

dataset = load_mimic_dataset(dataset_path)
existing_results = load_existing_results(results_path)

# Get the index of the last processed question
start_idx = len(existing_results)

# Initialize MedRAG model
medrag = MedRAG(llm_name="mistralai/Mixtral-8x7B-Instruct-v0.1", rag=True, retriever_name="MedCPT", corpus_name="StatPearls")

# Convert existing results to a DataFrame for easier appending
df = pd.DataFrame(existing_results)

for idx, (context, question, reference_answers) in enumerate(dataset[start_idx:], start=start_idx):
    # Generate answer
    generated_answer = medrag.answer(question=question, context=context, options=None, k=32)

    if reference_answers:
        reference_answer = reference_answers[0]  # Assuming we use the first reference answer for simplicity

    # Create a new DataFrame for the current result
    new_data = pd.DataFrame([{
        "idx": idx,
        "question": question,
        "reference_answer": reference_answer,
        "complete_answer": generated_answer,
    }])

    # Concatenate the new data with the existing DataFrame
    df = pd.concat([df, new_data], ignore_index=True)

    # Save results incrementally
    df.to_csv(results_path, index=False)

    print("=" * 50)
    print("Question:", question)
    print("/" * 50)
    print("Complete Answer:", generated_answer)
    print("|" * 50)
    print("Reference Answer:", reference_answer)
    print(idx)
    print("=" * 50)

print("Data saved to mimic_answers.csv")

