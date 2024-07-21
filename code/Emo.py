import numpy as np

def get_completion(client, s_prompt, u_prompt, model = "gpt-3.5-turbo-0125", temperature =0, tp = 'text'):
  messages = [{"role":"system", "content": s_prompt}, {"role":"user", "content": u_prompt}]
  response = client.chat.completions.create(
      model = model,
      messages = messages,
      response_format={"type": tp},
      temperature =temperature # degree of expiration (randomness) (0 same result - 1 creative)
  )
  return response.choices[0].message.content

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

def get_social_emo(client, d):
  code = {'POSITIVE': 1, 'NEUTRAL':0, 'NEGATIVE':-1}

  start = 0
  end = 1
  neg_timeline = []
  idx = []
  for i in range(len(d)):
    if d['Utterance end time (milliseconds)'].iloc[i] > (end*d['Utterance end time (milliseconds)'].iloc[-1]/20):
      c = 0
      for j in d['Sentiment'].iloc[start:i+1]:
        if j == 'NEGATIVE':
          c += 1
      neg_timeline.append(c/len(d))
      idx.append([start,i+1])
      start = i+1
      end = end+1
    c=0
  for j in d['Sentiment'].iloc[start:len(d)-1]:
      if j == 'NEGATIVE':
        c += 1
  neg_timeline.append(c/len(d))

  s_final = []
  t_final = []
  l = []
  s = []
  start = 0
  end = 1
  for i in range(len(d)):
    if d['Utterance end time (milliseconds)'].iloc[i] > (end*60000):
      l.append(list(d['Sentiment'].iloc[start:i+1]))
      s.append(list(d['Speaker'].iloc[start:i+1]))
      start = i+1
      end = end+1
  l.append(list(d['Sentiment'].iloc[start:len(d)-1]))
  s.append(list(d['Speaker'].iloc[start:len(d)-1]))

  for i in range(len(l)):
    stu = []
    tu = []
    for k in range(len(l[i])):
      if s[i][k] == 'student':
        stu.append(l[i][k])
      else:
        tu.append(l[i][k])
    s_final.append(sum([code[x]for x in stu]))
    t_final.append(sum([code[x]for x in tu]))

  x = list(range(len(s_final))) # s_final~x, t_final~x

  prompt_s = f"""You are an expert educator, I will provide you two list of student and tutor sentiment changes. Your task is to write a short summary for that. """
  prompt_u = f"""Here is the sentiment list for student {s_final} and here is the sentiment list for tutor {t_final}.
  Each element in the list represent the sum of sentiment of that person in one minite. Positive has a value of 1, Negative has a value of -1 and Neutral has a value of 0.
  Write a summary about 50 words to decribe what you got. """
  r = get_completion(client, prompt_s, prompt_u)

  total_social_emo_score = (sum(t_final)+sum(s_final))/(len(s_final)+len(t_final))
  return(normalize_timeline(neg_timeline), s_final,t_final,x, r, total_social_emo_score)

