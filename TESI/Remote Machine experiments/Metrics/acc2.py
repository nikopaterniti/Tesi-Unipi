import pandas as pd

# Carica il dataset
df = pd.read_csv('pima_predictions_clean.csv')

#df['class'].fillna(df['outcome'], inplace=True)

# Confronta le previsioni con i valori reali
correct_predictions = df['class'] == df['outcome']

# Calcola l'accuratezza
accuracy = correct_predictions.mean()
print(f'Accuracy: {accuracy * 100:.2f}%')

