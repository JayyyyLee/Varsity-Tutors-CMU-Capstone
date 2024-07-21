from collections import Counter
import numpy as np

def normalize_timeline(numbers):
  # Normalize to range 0-1
  min_val = min(numbers)
  max_val = max(numbers)
  normalized_numbers = (np.array(numbers) - min_val) / (max_val - min_val)
  # Scale to range 0-4
  scaled_numbers = normalized_numbers * 4
  # Round to nearest integer
  regularized_numbers = np.round(scaled_numbers).astype(int)
  return regularized_numbers.tolist()

def get_completion(client, s_prompt, u_prompt, model = "gpt-3.5-turbo-0125", temperature =0, tp = 'text'):
  messages = [{"role":"system", "content": s_prompt}, {"role":"user", "content": u_prompt}]
  response = client.chat.completions.create(
      model = model,
      messages = messages,
      response_format={"type": tp},
      temperature =temperature # degree of expiration (randomness) (0 same result - 1 creative)
  )
  return response.choices[0].message.content


def get_feedback(client, data):
  counter = Counter(data[(data['Speaker'] == 'tutor') & (data['DA'] == 'Feedback')]['Sentiment'])
  total_count = sum(counter.values())
  percentage_dict = {key: (value / total_count) * 100 for key, value in counter.items()}

  start = 0
  end = 1
  bad_neg_l = []
  neg_l = []
  bad_pos_l = []
  pos_l = []
  idx = []
  for i in range(len(data)):
    if data['Utterance end time (milliseconds)'].iloc[i] > (end*data['Utterance end time (milliseconds)'].iloc[-1]/20):
      neg_count = 0
      b_neg = 0
      pos_count = 0
      b_pos = 0
      for j in range(start, i+1):
        if (data.iloc[j]['Speaker'] == 'tutor') & (data.iloc[j]['DA'] == 'Feedback')& (data.iloc[j]['Sentiment'] == 'NEGATIVE'):
          neg_count +=1
          if i+4 < len(data):
            temp = data.iloc[i:i+4]['DA']
            if ('Explanation' not in list(temp)):
              b_neg += 1
        if (data.iloc[j]['Speaker'] == 'tutor') & (data.iloc[j]['DA'] == 'Feedback')& (data.iloc[j]['Sentiment'] == 'POSITIVE'):
          pos_count +=1

          prompt_s = f"""You are an expert educator, I will provide you positive feedback from a tutor to a student"""
          prompt_u = f"""Here is the feedback from the tutor: '{data.iloc[j]['Utterance']}'.
          If the tutor is praising student's effort or encouraging the student return 1, otherwise return 0. only return 0 or 1 """
          r = get_completion(client, prompt_s, prompt_u, model = 'gpt-4o')

          if r == '0':
            b_pos += 1
          if r == 0:
            b_pos += 1

      bad_neg_l.append(b_neg)
      neg_l.append(neg_count)

      bad_pos_l.append(b_pos)
      pos_l.append(pos_count)

      idx.append([start,i+1])
      start = i+1
      end = end+1

  neg_count = 0
  b_neg = 0
  pos_count = 0
  b_pos = 0
  for j in range(start, i+1):
    if (data.iloc[j]['Speaker'] == 'tutor') & (data.iloc[j]['DA'] == 'Feedback')& (data.iloc[j]['Sentiment'] == 'NEGATIVE'):
      neg_count +=1
      if i+4 < len(data):
        temp = data.iloc[i:i+4]['DA']
        if ('Explanation' not in list(temp)):
          b_neg += 1
    if (data.iloc[j]['Speaker'] == 'tutor') & (data.iloc[j]['DA'] == 'Feedback')& (data.iloc[j]['Sentiment'] == 'POSITIVE'):
      pos_count +=1

      prompt_s = f"""You are an expert educator, I will provide you positive feedback from a tutor to a student"""
      prompt_u = f"""Here is the feedback from the tutor: '{data.iloc[j]['Utterance']}'.
      If the tutor is praising student's effort or encouraging the student return 1, otherwise return 0. only return 0 or 1 """
      r = get_completion(client, prompt_s, prompt_u, model = 'gpt-4o')

      if r == '0':
        b_pos += 1
      if r == 0:
        b_pos += 1

  bad_neg_l.append(b_neg)
  neg_l.append(neg_count)

  bad_pos_l.append(b_pos)
  pos_l.append(pos_count)

  if sum(neg_l) ==0:
    neg_p = -1
  else:
    neg_p = sum(bad_neg_l)/sum(neg_l)
  if sum(pos_l) ==0:
    pos_p = -1
  else:
    pos_p = sum(bad_pos_l)/sum(pos_l)
  feedback_score = neg_p + pos_p

  tl = [(bad_neg_l[i] / neg_l[i] if neg_l[i] != 0 else 0) + (bad_pos_l[i] / bad_pos_l[i] if bad_pos_l[i] != 0 else 0) for i in range(len(bad_pos_l))]
  tl = normalize_timeline(tl)
  return (percentage_dict, tl, neg_p, pos_p, feedback_score)