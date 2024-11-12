import os
import torch
import json
import pandas as pd
from src.medrag import MedRAG
import re

# SELEZIONA GPU
gpu_ids = "1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il dataset
df = pd.read_csv('X_test.csv')
output_csv_file = 'X_test_answers.csv'

# Funzione per creare una descrizione testuale per ogni paziente
def create_patient_description(row):
    description = (f"The patient is {row['Age']} years old, "
                   f"has had {row['Pregnancies']} pregnancies, "
                   f"has a glucose level of {row['Glucose']}, "
                   f"blood pressure of {row['BloodPressure']}, "
                   f"a skin thickness of {row['SkinThickness']}, "
                   f"an insulin level of {row['Insulin']}, "
                   f"a BMI of {row['BMI']}, "
                   f"a diabetes pedigree function of {row['DiabetesPedigreeFunction']}.")
    return description

def create_patient_question(row):
    description = create_patient_description(row)
    question = f"Based on the following information: {description}, does the patient have diabetes?"
    return question

# Funzione per estrarre la classificazione dall'answer
def extract_classification(answer):
    # Usa una regex per cercare 'classification' seguito da ': 0' o ': 1'
    match = re.search(r'classification["\']?\s*:\s*(\d)', answer)
    if match:
        return int(match.group(1))  # Restituisce 0 o 1 come intero
    return None  # Se non trova nulla, ritorna None

# Inizializza MedRAG
medrag = MedRAG(llm_name="mistralai/Mixtral-8x7B-Instruct-v0.1", rag=True, retriever_name="MedCPT", corpus_name="StatPearls")

if not os.path.exists(output_csv_file):
    columns = ['idx','class','outcome', 'answer', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
               'BMI', 'DiabetesPedigreeFunction', 'Age']
    pd.DataFrame(columns=columns).to_csv(output_csv_file, index=False)

# Ciclo su ogni paziente per fare la predizione
for index, row in df.iterrows():

    question = create_patient_question(row)

    # Chiedi a MedRAG la risposta
    answer = medrag.answer(question=question, options=None, k=10)

    classification = extract_classification(answer)

    print(f"Patient {index + 1}:")
    print(answer)
    print(f"Classification: {classification}")
    print("-" * 50)

    # Crea un dataframe temporaneo per questa predizione
    temp_df = pd.DataFrame([{
        'idx': index,
        'class': classification,
        'outcome': row['Outcome'],
        'answer': answer,
        'Pregnancies': row['Pregnancies'],
        'Glucose': row['Glucose'],
        'BloodPressure': row['BloodPressure'],
        'SkinThickness': row['SkinThickness'],
        'Insulin': row['Insulin'],
        'BMI': row['BMI'],
        'DiabetesPedigreeFunction': row['DiabetesPedigreeFunction'],
        'Age': row['Age']
    }])

    temp_df.to_csv(output_csv_file, mode='a', header=False, index=False)


'''
# Aggiungi le predizioni al dataframe
df['idx'] = [pred['idx'] for pred in predictions]
df['answer'] = [pred['answer'] for pred in predictions]

# Salva il dataframe con le predizioni in un file CSV
output_csv_file = 'pima_predictions.csv'
df.to_csv(output_csv_file, index=False)
print(f"Results saved to {output_csv_file}")
'''
