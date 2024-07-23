import json

def get_completion(client, s_prompt, u_prompt, model = "gpt-3.5-turbo-0125", temperature =0, tp = 'text'):
  messages = [{"role":"system", "content": s_prompt}, {"role":"user", "content": u_prompt}]
  response = client.chat.completions.create(
      model = model,
      messages = messages,
      response_format={"type": tp},
      temperature =temperature # degree of expiration (randomness) (0 same result - 1 creative)
  )
  return response.choices[0].message.content

def tech(client, data):
  duration = data['Utterance end time (milliseconds)'].iloc[-1]
  start = 0
  end = 1
  l = []
  idx = []
  for i in range(len(data)):
    if data['Utterance end time (milliseconds)'].iloc[i] > (end*data['Utterance end time (milliseconds)'].iloc[-1]/20):
        d = ' '.join(data['Speaker'][start:i+1] +': ' +data['Utterance'][start:i+1])
        prompt_s = f"""You are an expert educator, I will provide you a chunk of transcript between a student and a tutor in a online one-on-one tutoring session.
        You task is to identidy that whether the tutor is experiencing a technological issue."""
        prompt_u = f"""Here is the transctipt {d}. If they are experiencing any of the following situations output 1:
        do not know how to use a tool, cannot hear each other, cannot see each other, cannot upload a file, cannot share the screen, or cannot see the screen.
        otherwise output 0. Format your answer as a json with 0 or 1 as key, and reasoning as valuing, summarize the reasoning in 10 words"""
        r = get_completion(client, prompt_s, prompt_u, model = 'gpt-4o', tp = "json_object")

        l.append(int(list(json.loads(r).keys())[0])*2)
        idx.append([start,i+1])
        start = i+1
        end = end+1
  d = ' '.join(data['Speaker'][start:len(data)-1] +': ' +data['Utterance'][start:len(data)-1])
  prompt_s = f"""You are an expert educator, I will provide you a chunk of transcript between a student and a tutor in a online one-on-one tutoring session.
        You task is to identidy that whether the tutor is experiencing a technological issue with the plateform."""
  prompt_u = f"""Here is the transctipt {d}. If they are experiencing any of the following situations output 1:
        do not know how to use a tool, cannot hear each other, cannot see each other, cannot upload a file, cannot share the screen, or cannot see the screen.
        otherwise output 0. Format your answer as a json with 0 or 1 as key, and reasoning as valuing, summarize the reasoning in 10 words"""
  r = get_completion(client, prompt_s, prompt_u, model = 'gpt-4o', tp = "json_object")

  l.append(int(list(json.loads(r).keys())[0])*2)
  idx.append([start,i+1])

  total_tech_score = sum(list(map(int, l)))/len(l)
  return(l, idx, total_tech_score)