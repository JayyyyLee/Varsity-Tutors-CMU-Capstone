import pandas as pd
import numpy as np
from openai import OpenAI
import os

from session_general import get_topic, get_expectation, get_inactive, get_low_interaction_reason, get_interaction, get_session_general_summary

from Instruction import get_instruction
from Feedback import get_feedback
from Tech import tech
from Emo import get_social_emo



def get_completion(s_prompt, u_prompt, model = "gpt-3.5-turbo-0125", temperature =0, tp = 'text'):
  messages = [{"role":"system", "content": s_prompt}, {"role":"user", "content": u_prompt}]
  response = client.chat.completions.create(
      model = model,
      messages = messages,
      response_format={"type": tp},
      temperature =temperature # degree of expiration (randomness) (0 same result - 1 creative)
  )
  return response.choices[0].message.content

def get_tutor_performance_summary(i,t,f,n):
  prompt_s = f"""You are an expert educator, I will provide you some descriptive information about a one-on-one tutoring session between a tutor and student,
  your task is to summarize the information provided"""
  prompt_u = f"""We divide the session into 20 time slots, and we have 4 criterias for each time slot: instruction delivery, tech & tool usage, feedback quality, social-emotional teaching.
  Larger number indicates that there might be more problems in this time slot according to the criterias.
  Here are the scores for the 20 time slots for instruction delivery: {i}, 
  tech & tool usage: {t}, 
  feedback quality: {f} , 
  social-emotional teaching: {n}.
  While summarizeing, do not mention slot number; instead, refer to the relative position in the session. Format your output as a paragraph of about 50 words"""
  r = get_completion(prompt_s, prompt_u)
  return(r)

def calculate_percentiles(scores):
    sorted_scores = sorted(scores)
    percentiles = [round((sorted_scores.index(score) / len(sorted_scores)) * 100,1) for score in scores]
    return percentiles

def normalize_score(numbers):
  min_val = min(numbers)
  max_val = max(numbers)
  normalized_numbers = (np.array(numbers) - min_val) / (max_val - min_val)

  scaled_numbers = normalized_numbers * 9+1

  regularized_numbers = np.round(scaled_numbers).astype(int)

  return regularized_numbers.tolist()



if __name__ == '__main__':

    open_api_key = os.getenv('OPENAI_KEY')  # Retrieves the API key from the environment variable
    client = OpenAI(
      api_key= open_api_key 
    )
    
    names = os.listdir('DA')
    out = pd.read_csv('150df.csv')
    data = pd.read_csv('DA/' +names[0])

    topic = get_topic(client, data)
    expectation = get_expectation(client, data)
    low_inter_reason, low_inter_time = get_low_interaction_reason(client, data)
    inactive_time = get_inactive(data)
    s_inter,t_inter, x_inter = get_interaction(data)
    ai_session_sum = get_session_general_summary(client, expectation, topic, low_inter_reason, low_inter_time)

    instru_l, effective_score, align_p, factually_p = get_instruction(client, data)
    tech_l, idx, total_tech_score = tech(client, data)
    percentage_dict, feedback_l, neg_p, pos_p, feedback_score = get_feedback(client, data)
    neg_timeline, s_emo, t_emo, x_emo, words, total_social_emo_score = get_social_emo(client, data)

    dic = {
      'uid': names[0][:-4], 
      'topic': topic, 
      'expectation': expectation,
      'low_inter_reason': low_inter_reason, 'low_inter_time': low_inter_time,
      's_inter': s_inter,  't_inter': t_inter, 'x_inter': x_inter,
      'ai_session_sum': ai_session_sum,
      'instru_l': instru_l, 
      'effective_score': effective_score, 
      'align_p': align_p, 
      'factually_p': factually_p,
      'tech_l': tech_l, 
      'idx': idx, 
      'total_tech_score': total_tech_score,
      'percentage_dict': percentage_dict, 
      'feedback_l': feedback_l, 
      'neg_p': neg_p, 'pos_p': pos_p, 
      'feedback_score': feedback_score,
      'neg_timeline': neg_timeline, 
      's_emo': s_emo, 't_emo': t_emo, 'x_emo': x_emo, 
      'words': words, 
      'total_social_emo_score': total_social_emo_score, 
      'tutor_performance_summary': get_tutor_performance_summary(instru_l, tech_l, feedback_l, neg_timeline),
      'final_tech_score': 0, 'final_feedback_score': 0, 'final_emo_score': 0, 'final_instru_score': 0,
      'final_tech_all_p': 0, 'final_feedback_all_p': 0, 'final_emo_all_p': 0, 'final_instru_all_p': 0,
      'time': data['Utterance end time (milliseconds)'].iloc[-1], 'inactive' :inactive_time
    }

    out.loc[len(out)] = dic

    df = out
      
    df['final_tech_score'] = normalize_score(list(1- df['total_tech_score']))
    df['final_feedback_score'] = normalize_score(list(2- df['feedback_score']))
    df['final_emo_score'] = normalize_score(df['total_social_emo_score'])
    df['final_instru_score'] = normalize_score(df['factually_p'] + df['align_p'] +df['effective_score'] )

    df['final_tech_all_p'] = calculate_percentiles(df['final_tech_score'])
    df['final_feedback_all_p'] = calculate_percentiles(df['final_feedback_score'])
    df['final_emo_all_p'] = calculate_percentiles(df['final_emo_score'])
    df['final_instru_all_p'] = calculate_percentiles(df['final_instru_score'])

  
    df.to_csv('150df.csv', index=False)
