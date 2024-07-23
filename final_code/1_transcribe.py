import pandas as pd
import assemblyai as aai
import json
from openai import OpenAI
import os

def get_completion(prompt, model = "gpt-4-turbo-preview", temperature =0):
  messages = [{"role":"user", "content": prompt}]
  response = client.chat.completions.create(
      model = model,
      messages = messages,
      temperature =temperature # degree of expiration (randomness) (0 same result - 1 creative)
  )
  return response.choices[0].message.content

def get_words(transcript):
  speaker = []
  words = []
  start_time = []
  end_time = []
  sentiment = []
  for utterance in transcript.sentiment_analysis:
    speaker.append(utterance.speaker)
    words.append(utterance.text)
    start_time.append(utterance.start)
    end_time.append(utterance.end)
    sentiment.append(utterance.sentiment)
  result = pd.DataFrame({'Speaker': speaker, 'Utterance start time (milliseconds)': start_time, 'Utterance end time (milliseconds)': end_time,  'Utterance': words, 'Sentiment': sentiment})
  return result

def transcribe(in_path, out_path):
  transcript = aai.Transcriber().transcribe(in_path, config)
  if transcript.utterances is None:
    result = pd.DataFrame({'Speaker': [], 'Utterance start time (milliseconds)': [], 'Utterance end time (milliseconds)': [],  'Utterance': [], 'Sentiment':[]})
  else:
    result = get_words(transcript)
  if len(result) != 0:

    temp = [result['Speaker'][i]+ " : " + result['Utterance'][i]  for i in range(len(result))]
    d = ' '.join(temp) [:900]
    prompt = f"""I will provide you a transcript of a tutor and a student.
    There will to two speakers A and B. You task is to identify which is student and which is tutor.
    In this session, a tutor may teach students knowledge about the topic, help student solve their problem, or ask questions to check the students' understanding.
    Format your output in a JSON format with A,B as key; student, tutor as value
    """ + d
    r = get_completion(prompt)
    a = json.loads(r[8:-4])
    result = result.replace({'Speaker': a})
    result.to_csv(out_path+in_path[-40:-4]+'.csv', index=False)
  else:
    result.to_csv(out_path+in_path[-40:-4]+'.csv', index=False)

if __name__ == '__main__':
    
    assemb_api_key = os.getenv('ASSEMB_KEY')
    aai.settings.api_key = assemb_api_key
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        sentiment_analysis = True,
        auto_chapters = True,
        disfluencies=True
    )

    open_api_key = os.getenv('OPENAI_KEY')  # Retrieves the API key from the environment variable
    client = OpenAI(
      api_key= open_api_key 
    )


    names = os.listdir('audio')
    names = sorted(names)
    os.makedirs('data/')
    transcribe('audio/' + names[0], 'data/')

