import pandas as pd
import numpy as np
from openai import OpenAI
import ast
from json_format import jsondict
import json

def get_completion(s_prompt, u_prompt, model = "gpt-3.5-turbo-0125", temperature =0, tp = 'text'):
  messages = [{"role":"system", "content": s_prompt}, {"role":"user", "content": u_prompt}]
  response = client.chat.completions.create(
      model = model,
      messages = messages,
      response_format={"type": tp},
      temperature =temperature # degree of expiration (randomness) (0 same result - 1 creative)
  )
  return response.choices[0].message.content

def str_to_list(s):
    return ast.literal_eval(s)

def sum_lists(row):
    return [a + b for a, b in zip(row['s_inter'], row['t_inter'])]

def silence(row):
  return sum([i<3 for i in row['silence']]) / len(row['silence'])

def stu_silence(row):
  return sum([i<3 for i in row['s_inter']]) / len(row['s_inter'])


def calculate_percentiles(scores):
    sorted_scores = sorted(scores)
    percentiles = [round((sorted_scores.index(score) / len(sorted_scores)) * 100,1) for score in scores]
    return percentiles

if __name__ == '__main__':

  #open_api_key = os.getenv('OPENAI_KEY')  # Retrieves the API key from the environment variable
  client = OpenAI(
      api_key= "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"#open_api_key 
  )

  df = pd.read_csv('150df.csv')

  df['s_inter'] = df['s_inter'].apply(str_to_list)
  df['t_inter'] = df['t_inter'].apply(str_to_list)
  df['x_inter'] = df['x_inter'].apply(str_to_list)
  df['silence'] = df.apply(sum_lists, axis=1)
  df['silence'] = df.apply(silence, axis=1)
  df['silence_p'] = calculate_percentiles(df['silence'])

  df['s_silence'] = df.apply(stu_silence, axis=1)
  df['s_silence_p'] = calculate_percentiles(df['s_silence'])

  df['uid'] = [i[:-4] for i in df['uid']]
  info = pd.read_csv('session_info.csv')
  df = df.merge(info, how = 'left', left_on='uid', right_on = 'session_uid')
    
  df['final_tech_tutor_p'] = df.groupby('Tutor ID')['final_tech_score'].rank(pct=True)
  df['final_feedback_tutor_p'] = df.groupby('Tutor ID')['final_feedback_score'].rank(pct=True)
  df['final_emo_tutor_p'] = df.groupby('Tutor ID')['final_emo_score'].rank(pct=True)
  df['final_instru_tutor_p'] = df.groupby('Tutor ID')['final_instru_score'].rank(pct=True)

  temp_d = df

  sessionlist = []
  tutorlist = []
  for i in range(len(temp_d)):

    dl,sd,td, tl = jsondict(temp_d, i)
    sessionlist.append(dl)
    tutorlist.append(tl)
    with open("json/sessiondetail/sd" + temp_d.iloc[i]['uid'] + ".json", "w") as outfile:
      outfile.write(json.dumps(sd, indent=2))
    with open("json/tutordetail/td"+ temp_d.iloc[i]['uid'] +".json", "w") as outfile:
      outfile.write(json.dumps(td, indent=2))
  with open("json/sessionList.json", "w") as outfile:
    outfile.write(json.dumps(sessionlist, indent=2))
  with open("json/tutorList.json", "w") as outfile:
    outfile.write(json.dumps(tutorlist, indent=2))





