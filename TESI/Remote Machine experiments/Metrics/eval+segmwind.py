import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from tabulate import tabulate

# Scarica i tokenizzatori di NLTK
nltk.download('punkt')

# Funzione per calcolare il BLEU score
def calculate_bleu(reference, candidate):
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing_function)
    return bleu_score

# Funzione per calcolare il SacreBLEU score
def calculate_sacrebleu(reference, candidate):
    reference_list = [reference]  # sacrebleu expects a list of references
    candidate_list = [candidate]
    bleu_score = corpus_bleu(candidate_list, [reference_list])
    return bleu_score.score / 100  # sacrebleu returns a score out of 100

# Funzione per calcolare il ROUGE score
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Funzione per segmentare la frase usando una finestra scorrevole e calcolare il BLEU, SacreBLEU e ROUGE score per ciascun segmento
def segment_and_calculate_scores(reference, answer):
    reference_tokens = nltk.word_tokenize(reference)
    answer_tokens = nltk.word_tokenize(answer)
    window_size = len(reference_tokens)

    bleu_scores = []
    sacrebleu_scores = []
    rouge_scores = []

    # Genera segmenti usando una finestra scorrevole
    for i in range(len(answer_tokens) - window_size + 1):
        segment = answer_tokens[i:i + window_size]
        segment_str = ' '.join(segment)

        # Calcola il BLEU score per il segmento
        bleu_score = calculate_bleu(reference, segment_str)
        bleu_scores.append(bleu_score)

        # Calcola il SacreBLEU score per il segmento
        sacrebleu_score = calculate_sacrebleu(reference, segment_str)
        sacrebleu_scores.append(sacrebleu_score)

        # Calcola il ROUGE score per il segmento
        rouge_score = calculate_rouge(reference, segment_str)
        rouge_scores.append(rouge_score)

    # Restituisce il BLEU score più alto tra i segmenti
    max_bleu_score = max(bleu_scores) if bleu_scores else 0

    # Restituisce il SacreBLEU score più alto tra i segmenti
    max_sacrebleu_score = max(sacrebleu_scores) if sacrebleu_scores else 0

    # Restituisce il ROUGE score più alto tra i segmenti per ogni metrica
    max_rouge_scores = {
        'rouge1': max([score['rouge1'].fmeasure for score in rouge_scores]) if rouge_scores else 0,
        'rouge2': max([score['rouge2'].fmeasure for score in rouge_scores]) if rouge_scores else 0,
        'rougeL': max([score['rougeL'].fmeasure for score in rouge_scores]) if rouge_scores else 0,
    }

    return max_bleu_score, max_sacrebleu_score, max_rouge_scores

# Carica il dataset
file_path = 'processed_mimic+rag_answers.csv'
df = pd.read_csv(file_path)
columns_to_display = ['Question', 'Reference_Answer', 'Generated_Answer', 'Summary']
print(df[columns_to_display].head(10))

# Trasforma le colonne in stringhe per evitare problemi di tipo di dati
df['Reference_Answer'] = df['Reference_Answer'].astype(str)
df['Generated_Answer'] = df['Generated_Answer'].astype(str)

# Calcola i BLEU, SacreBLEU e ROUGE score per ogni risposta segmentata
results = df.apply(lambda row: segment_and_calculate_scores(row['Reference_Answer'], row['Generated_Answer']), axis=1)
df['max_bleu_score'] = results.apply(lambda x: x[0])
df['max_sacrebleu_score'] = results.apply(lambda x: x[1])
df['max_rouge1_score'] = results.apply(lambda x: x[2]['rouge1'])
df['max_rouge2_score'] = results.apply(lambda x: x[2]['rouge2'])
df['max_rougeL_score'] = results.apply(lambda x: x[2]['rougeL'])

# Filtra le colonne per visualizzare solo gli score
scores_df = df[['max_bleu_score', 'max_sacrebleu_score', 'max_rouge1_score', 'max_rouge2_score', 'max_rougeL_score']]

# Calcola i punteggi medi
average_max_bleu = scores_df['max_bleu_score'].mean()
average_max_sacrebleu = scores_df['max_sacrebleu_score'].mean()
average_max_rouge1 = scores_df['max_rouge1_score'].mean()
average_max_rouge2 = scores_df['max_rouge2_score'].mean()
average_max_rougeL = scores_df['max_rougeL_score'].mean()

# Stampa i risultati
print(tabulate(scores_df, headers='keys', tablefmt='psql'))
print(f"Average Max BLEU Score: {average_max_bleu:.4f}")
print(f"Average Max SacreBLEU Score: {average_max_sacrebleu:.4f}")
print(f"Average Max ROUGE-1 Score: {average_max_rouge1:.4f}")
print(f"Average Max ROUGE-2 Score: {average_max_rouge2:.4f}")
print(f"Average Max ROUGE-L Score: {average_max_rougeL:.4f}")
