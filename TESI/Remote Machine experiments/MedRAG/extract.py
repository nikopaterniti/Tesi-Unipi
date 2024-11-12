import json
import pandas as pd

def extract_text_after_keywords(predicted_answer, keywords):
    text_after_keywords = {keyword: "" for keyword in keywords}
    for keyword in keywords:
        keyword_index = predicted_answer.rfind(keyword)
        if keyword_index != -1:
            start_index = keyword_index + len(keyword) + 3  # Adjust for ": " after the keyword
            end_index = predicted_answer.find('}', start_index)
            text_after_keywords[keyword] = predicted_answer[start_index:end_index].strip(' "\'}')
    return text_after_keywords

# Function to process the DataFrame and extract relevant text
def process_predictions(df):
    keywords = ["step_by_step_thinking", "summary"]
    results = []

    for index, row in df.iterrows():
        question = row['question']
        reference_answer = row['reference_answer']
        predicted_answer = row['complete_answer']
        extracted_text = extract_text_after_keywords(predicted_answer, keywords)
        
        result_row = {
            "Question": question,
            "Generated_Answer": extracted_text["step_by_step_thinking"],
            "Summary": extracted_text["summary"],
            "Reference_Answer": reference_answer
        }
        results.append(result_row)
    
    return pd.DataFrame(results)

# Load your data
output_file = 'mimic_answers.csv'
df = pd.read_csv(output_file)

# Process the data to extract text after keywords
processed_df = process_predictions(df)

# Save the processed data to a new CSV file
processed_output_file = 'processed_mimic+rag_answers.csv'
processed_df.to_csv(processed_output_file, index=False)

print(f"Processed data saved to {processed_output_file}")

# Optional: Display the first few rows of the processed DataFrame
print(processed_df.head())
