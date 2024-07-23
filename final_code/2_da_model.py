import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def prepare_data(sentences):
    encodings = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return encodings

def predict(model, sentences):
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # disable gradient computation for inference
        inputs = prepare_data(sentences)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

def da_classify(in_path, out_path, model):
  d = pd.read_csv(in_path)
  new_sentences = list(d['Speaker']+ ': ' +d['Utterance'])

  label_dict = {'Extra Domain': 0, 'Questions': 1, 'Other': 2, 'Explanation': 3, 'Feedback': 4}

  predicted_labels = predict(model, new_sentences)

  label_names = {value: key for key, value in label_dict.items()}  # Assuming label_dict from training
  predicted_label_names = [label_names[label.item()] for label in predicted_labels]

  d['DA'] = predicted_label_names
  d.to_csv(out_path, index=False)


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("J4YL19/CAPSTONE_DA")
    model = AutoModelForSequenceClassification.from_pretrained("J4YL19/CAPSTONE_DA")

    names = os.listdir('data')
    os.makedirs('DA/')
    da_classify('data/' + names[0], 'DA/' +names[0]+'.csv', model)

