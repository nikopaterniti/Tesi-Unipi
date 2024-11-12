import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tabulate import tabulate

# Assicurati di avere il pacchetto nltk 'punkt' scaricato
nltk.download('punkt')

# Funzione per calcolare il BLEU score con smoothing
def calculate_bleu(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)
    return bleu_score

# Funzione per calcolare il ROUGE score
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Leggi il file CSV
#csv_file = 'mimic_test.csv'
csv_file = 'processed_mimic+rag_answers.csv'
#csv_file = 'mimic_answers_short2.csv'
df = pd.read_csv(csv_file)

#df['answer'] = df['answer'].astype(str)

# Inizializza le liste per memorizzare i risultati
bleu_scores = []
rouge_scores = []
references = []
candidates = [] 
# Calcola le metriche per ogni coppia di risposta
for index, row in df.iterrows():
    reference_answer = row['Reference_Answer']
    generated_answer = row['Summary']

    # Calcola BLEU score
    bleu = calculate_bleu(reference_answer, generated_answer)
    bleu_scores.append(bleu)

    # Calcola ROUGE score
    rouge = calculate_rouge(reference_answer, generated_answer)
    rouge_scores.append(rouge)

    # Aggiungi risposte alle liste per calcolo BERTScore
    references.append(reference_answer)
    candidates.append(generated_answer)

# Calcola BERTScore
P, R, F1 = bert_score(candidates, references, lang='en', verbose=True)

# Aggiungi i risultati al DataFrame
df['bleu_score'] = bleu_scores
df['rouge1'] = [score['rouge1'].fmeasure for score in rouge_scores]
df['rouge2'] = [score['rouge2'].fmeasure for score in rouge_scores]
df['rougeL'] = [score['rougeL'].fmeasure for score in rouge_scores]
df['bertscore_precision'] = P
df['bertscore_recall'] = R
df['bertscore_f1'] = F1

# Filtra le colonne per visualizzare solo gli score
scores_df = df[['bleu_score', 'rouge1', 'rouge2', 'rougeL', 'bertscore_precision', 'bertscore_recall', 'bertscore_f1']]

# Escludi gli score assoluti pari a zero
scores_df_non_zero = scores_df[(scores_df['bleu_score'] != 0) | (scores_df['rouge1'] != 0) | (scores_df['rouge2'] != 0) | (scores_df['rougeL'] != 0) | (scores_df['bertscore_precision'] != 0) | (scores_df['bertscore_recall'] != 0) | (scores_df['bertscore_f1'] != 0)]

# Calcola i punteggi medi escludendo gli score pari a zero
average_bleu = scores_df_non_zero['bleu_score'][scores_df_non_zero['bleu_score'] != 0].mean()
average_rouge1 = scores_df_non_zero['rouge1'][scores_df_non_zero['rouge1'] != 0].mean()
average_rouge2 = scores_df_non_zero['rouge2'][scores_df_non_zero['rouge2'] != 0].mean()
average_rougeL = scores_df_non_zero['rougeL'][scores_df_non_zero['rougeL'] != 0].mean()
average_bertscore_precision = scores_df_non_zero['bertscore_precision'][scores_df_non_zero['bertscore_precision'] != 0].mean()
average_bertscore_recall = scores_df_non_zero['bertscore_recall'][scores_df_non_zero['bertscore_recall'] != 0].mean()
average_bertscore_f1 = scores_df_non_zero['bertscore_f1'][scores_df_non_zero['bertscore_f1'] != 0].mean()

# Visualizza il DataFrame aggiornato in modo tabellare
print(tabulate(scores_df_non_zero, headers='keys', tablefmt='psql'))

# Visualizza i punteggi medi
#print("Average Scores (excluding absolute zeros):")
print(f"Average BLEU Score: {average_bleu:.4f}")
print(f"Average ROUGE-1 Score: {average_rouge1:.4f}")
print(f"Average ROUGE-2 Score: {average_rouge2:.4f}")
print(f"Average ROUGE-L Score: {average_rougeL:.4f}")
print(f"Average BERTScore Precision: {average_bertscore_precision:.4f}")
print(f"Average BERTScore Recall: {average_bertscore_recall:.4f}")
print(f"Average BERTScore F1: {average_bertscore_f1:.4f}")

# Opzionalmente, salva i risultati su un nuovo file CSV
df.to_csv('mimic_answers_metrics.csv', index=False)
