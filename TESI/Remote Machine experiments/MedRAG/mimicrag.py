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

dataset_path = 'physionet.org/files/mimic-iii-question-answer/1.0.0/test.final.json'
dataset = load_mimic_dataset(dataset_path)

medrag = MedRAG(llm_name="mistralai/Mixtral-8x7B-Instruct-v0.1", rag=True, retriever_name="MedCPT", corpus_name="StatPearls")

data = []

for idx, (context, question, reference_answers) in enumerate(dataset):
    
    # Generate answer
    generated_answer = medrag.answer(question=question, context=context, options=None, k=32)

    if reference_answers:
        reference_answer = reference_answers[0]  # Assuming we use the first reference answer for simplicity
        
    # Aggiungi i dati al dataframe
    data.append({
        "idx": idx,
        "question": question,
        "reference_answer": reference_answer,
        "complete_answer": generated_answer,
    })

    print("=" * 50)
    print("Question:", question)
    print("/" * 50)
    print("Complete Answer:", generated_answer)
    print("|" * 50)
    print("Reference Answer:", reference_answer)
    print(idx)
    print("=" * 50)

df = pd.DataFrame(data)
df.to_csv('mimic_answers.csv', index=False)
print("Data saved to mimic_answers.csv")
