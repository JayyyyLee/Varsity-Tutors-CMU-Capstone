import pandas as pd

import ast
from json_format import jsondict
import json


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

def remove_duplicates(dict_list):
    # Convert list of dicts to a set of tuples (to remove duplicates)
    seen = set()
    unique_dicts = []
    for d in dict_list:
        # Create a tuple from dictionary items to make it hashable
        tuple_representation = (d['tutorId'], d['tutor'], tuple(d['subjects']))
        if tuple_representation not in seen:
            seen.add(tuple_representation)
            unique_dicts.append(d)
    return unique_dicts

if __name__ == '__main__':

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
    with open("json/sessiondetail/sd_" + temp_d.iloc[i]['uid'] + ".json", "w") as outfile:
      outfile.write(json.dumps(sd, indent=2))
    with open("json/tutordetail/td_"+ temp_d.iloc[i]['Tutor ID'] +".json", "w") as outfile:
      outfile.write(json.dumps(td, indent=2))
  with open("json/sessionList.json", "w") as outfile:
    outfile.write(json.dumps(sessionlist, indent=2))

  tutorlist = remove_duplicates(tutorlist)
  with open("json/tutorList.json", "w") as outfile:
    outfile.write(json.dumps(tutorlist, indent=2))




