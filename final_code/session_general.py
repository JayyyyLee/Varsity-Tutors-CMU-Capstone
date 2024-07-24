from collections import Counter

def get_completion(client, s_prompt, u_prompt, model = "gpt-3.5-turbo-0125", temperature =0, tp = 'text'):
  messages = [{"role":"system", "content": s_prompt}, {"role":"user", "content": u_prompt}]
  response = client.chat.completions.create(
      model = model,
      messages = messages,
      response_format={"type": tp},
      temperature =temperature # degree of expiration (randomness) (0 same result - 1 creative)
  )
  return response.choices[0].message.content

def get_topic(client, d):
  text = ' '.join(d['Speaker'] +': '+ d['Utterance'])
  prompt_s = f"""You are an expert educator, I will provide you a transcript of a one-on-one tutoring session between a tutor and student,
  your task is to identify the topics they are solving in the session"""
  prompt_u = f"""Here is the transcript {text}, format your output as JSON format, using Question numbers as the keys.
  Here is a example json format: 'Question 1': 'Topic 1',  'Question 2': 'Topic 2', 'Question 3': 'Topic 3', 'Question 4': 'Topic 4'.
  Output at most 9 questions. """
  r = get_completion(client, prompt_s, prompt_u, tp = "json_object")
  return(r)

def get_expectation(client, d):
  text = ' '.join(d['Speaker'][:100] +': ' +d['Utterance'][:100])
  prompt_s = "I will provide you a transcript of the beginning a one-on-one tutoring session between a tutor and student, your task is to identify student's expectation for the session, and whether these expectations are solved"
  prompt_u = f"""Here is the transcript {text}, format your in a short paragraph of about 20 words"""
  r = get_completion(client, prompt_s, prompt_u)
  return (r)

def get_interaction(d):
  l = []
  start = 0
  end = 1
  for i in range(len(d)):
    if d['Utterance end time (milliseconds)'].iloc[i] > (end*60000):
      l.append(list(d['Speaker'].iloc[start:i+1]))
      start = i+1
      end = end+1
  l.append(list(d['Speaker'].iloc[start:len(d)-1]))
  s = [Counter(k)['student'] for k in l]
  t = [Counter(k)['tutor'] for k in l]
  x = list(range(len(t))) # s~x, t~x two lines

  return (s,t,x)


def get_low_interaction_reason(client, d):
  d = d.fillna('NA')
  l = []
  start = 0
  end = 1
  for i in range(len(d)):
    if d['Utterance end time (milliseconds)'].iloc[i] > (end*60000):
      l.append(list(d['Speaker'].iloc[start:i+1]))
      start = i+1
      end = end+1
  l.append(list(d['Speaker'].iloc[start:len(d)-1]))
  s = [Counter(k)['student'] for k in l]
  t = [Counter(k)['tutor'] for k in l]
  low = [a + b for a, b in zip(s, t)]

  ct = 0
  l = []
  for i in range(len(low)):
    if low[i] <=3:
      if ct == 0:
        temp = i
      ct +=1
    else:
      if ct >= 1:
        l.append([(max(temp-1, 0)),i])
      ct = 0

  temp = 0
  quote = ''
  result = []

  for i in range(len(d)):
    if temp >= len(l):
      break
    if (d.iloc[i]['Utterance start time (milliseconds)'] >= l[temp][0]):
      quote += d['Speaker'].iloc[i] + ':' + d['Utterance'].iloc[i] +'\n'
    if (d.iloc[i]['Utterance end time (milliseconds)'] >= l[temp][1]):

      prompt_s = f"""I will provide you a chunk of transcript from a one-on-one tutoring session between a tutor and student.
              This is a chunk before a period of silence during the session. """
      prompt_u = f"""Here is the transcript {quote}. Your task is to analyze the reason why the tutor and the student had a period of silence after this.
              Paraphrase the reason to 15 words and output. """
      r = get_completion(client, prompt_s, prompt_u)
      result.append(r)
      temp += 1
      quote = ''
  return(result, l)

def get_session_general_summary(client, expectation, topic, low_inter_reason, low_inter_time):
  prompt_s = f"""You are an expert educator, I will provide you some descriptive information about a one-on-one tutoring session between a tutor and student,
  your task is to summarize the information provided"""
  prompt_u = f"""Here are the topics they discussed in the session: {topic} , here is the student expectation at the beginning of the session: {expectation}.
  Here are some time periods when student and tutor had low interaction (in milliseconds): {low_inter_time},
  and the reasons why the tutor and the student had low interaction for each period: {low_inter_reason}.
  Format your output as a paragraph of about 50 words"""
  r = get_completion(client, prompt_s, prompt_u)
  return(r)