import pandas as pd

# Leggi i file CSV
pima_predictions = pd.read_csv('pima_predictions.csv')
diabetes = pd.read_csv('diabetes.csv')

# Estrai le colonne da confrontare
predictions = pima_predictions['class']
outcome = diabetes['Outcome']

#pima_predictions['class'] = predictions.fillna(outcome)

# Ora confrontiamo le predizioni (modificate) con l'outcome reale
if len(pima_predictions['class']) != len(outcome):
    print("Le due colonne hanno lunghezze diverse, non possono essere confrontate.")
else:
    # Confronto tra le colonne
    comparison = pima_predictions['class'] == outcome
    
    # Conta quante predizioni sono corrette
    correct_predictions = comparison.sum()
    total_predictions = len(pima_predictions['class'])
    accuracy = correct_predictions / total_predictions * 100

    # Stampa i risultati
    print(f"Predizioni corrette: {correct_predictions}/{total_predictions}")
    print(f"Accuratezza: {accuracy:.2f}%")

    # Opzionalmente, mostra le discrepanze
    discrepancies = pima_predictions[~comparison]
    if not discrepancies.empty:
        print("\nDiscrepanze trovate:")
        print(discrepancies[['idx', 'class']])

# Salva il nuovo CSV con i valori modificati
pima_predictions.to_csv('pima_predictions_modified.csv', index=False)
