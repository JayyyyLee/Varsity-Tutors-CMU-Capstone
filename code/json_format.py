import pandas as pd
import ast
import json

def gen_risk(s):
  risklist = []

  du = round(s['time']/ 60000, 1)
  d = s['Sessions duration (min)']

  if s['final_instru_all_p']<0.3:
    risklist.append('Instru')
  if s['final_tech_all_p']<0.3:
    risklist.append('tech')
  if s['final_feedback_all_p']<0.3:
    risklist.append('Feedback')
  if s['final_emo_all_p']<0.3:
    risklist.append('Emotion')
  if s['final_emo_all_p']<0.3:
    risklist.append('Emotion')
  if s['silence_p']>70:
    risklist.append('Silent')
  if s['s_silence_p']>70:
    risklist.append('Student Inactive')
  if du/d <0.8:
    risklist.append('Short session')

  return risklist

def jsondict(temp_d, i):
  s = temp_d.iloc[i]
  du = round(s['time']/ 60000, 1)

  rl = gen_risk(s)

  dl =   {
      "sessionId": s ['uid'],
      "tutorId": i,
      "tutor": s['Tutor ID'],
      "subject":s['subject'],
      "date":s['tutoring session occurred date'],
      "duration":str(du) + '/' + str(s['Sessions duration (min)']),
      "riskyAreas": rl,
      "instructionalDelivery":int( s['final_instru_score']),
      "technicalIssues": int(s['final_tech_score']),
      "feedbackQuality":int(s['final_feedback_score']),
      "socioEmotionalTeaching":int(s['final_emo_score']),
      "silentSessionPercent": s['silence_p'],
      "inactiveSessionPercent": s['s_silence_p']}

  sd =  {
    "sessionId": s ['uid'],
    "tutor":s['Tutor ID'],
    "student": int(s['student_id']),
    "subject":s['subject'],
    "date":s['tutoring session occurred date'],
    "riskyAreas": rl,
    "sessionGeneral": {
      "aiSummary": s['ai_session_sum'],
      "desiredOutcome": s['expectation'],
      "activities":list(ast.literal_eval(temp_d['topic'].iloc[i]).values()),
      "interactionTrend":{
        "tutorUtterances": s['t_inter'],
        "studentUtterances": s['s_inter'],
        "time":  s['x_inter'],
        "analysis": ast.literal_eval(temp_d['low_inter_reason'].iloc[i])
      }
    },
    "tutorPerformance":{
      "aiSummary": s['tutor_performance_summary'],
      "sessionTimeline":{
        "time":  [f"{i:02}" for i in range(0, round(du/20)*20+1, round(du/20))],
        "categories": [
          {
            "name": "Instruction Delivery",
            "score": int(s['final_instru_score']),
            "percentile": s['final_instru_all_p'],
            "values": ast.literal_eval(temp_d['instru_l'].iloc[i])
          },
          {
            "name": "Tech & Tool Usage",
            "score": int(s['final_tech_score']),
            "percentile": s['final_tech_all_p'],
            "values": ast.literal_eval(temp_d['tech_l'].iloc[i])
          },
          {
            "name": "Feedback Quality",
            "score": int(s['final_feedback_score']),
            "percentile": s['final_feedback_all_p'],
            "values": ast.literal_eval(temp_d['feedback_l'].iloc[i])
          },
          {
            "name": "Social-Emotional Teaching",
            "score": int(s['final_emo_score']),
            "percentile": s['final_emo_all_p'],
            "values": ast.literal_eval(temp_d['neg_timeline'].iloc[i])
          }
        ]
      },
      "rating": {
        "instructionalDelivery": {
          "rating": int(s['final_instru_score']),
          "allSessionRanking": s['final_instru_all_p'],
          "tutorSessionRanking": s['final_instru_tutor_p'],
          "correctExplanation": s['factually_p'] *100,
          "incorrectExplanation": (1- s['factually_p']) *100,
          "alignedExplanation": s['align_p'] *100 ,
          "unalignedExplanation": (1-s['align_p']) *100 ,
          "effectiveQuestions": s['effective_score'] *100,
          "ineffectiveQuestions": (1-s['effective_score']) *100
        },
        "techToolUsage": {
          "rating": int(s['final_tech_score']),
          "allSessionRanking": s['final_tech_all_p'],
          "tutorSessionRanking": s['final_tech_tutor_p']
        },
        "feedbackQuality": {
          "rating": int(s['final_feedback_score']),
          "allSessionRanking": s['final_feedback_all_p'],
          "tutorSessionRanking": s['final_feedback_tutor_p'],
          "positiveFeedback": ast.literal_eval(temp_d['percentage_dict'].iloc[i])['POSITIVE'] if 'POSITIVE' in ast.literal_eval(temp_d['percentage_dict'].iloc[i]) else 0,
          "neutralFeedback": ast.literal_eval(temp_d['percentage_dict'].iloc[i])['NEUTRAL'] if 'NEUTRAL' in ast.literal_eval(temp_d['percentage_dict'].iloc[i]) else 0,
          "negativeFeedback": ast.literal_eval(temp_d['percentage_dict'].iloc[i])['NEGATIVE'] if 'NEGATIVE' in ast.literal_eval(temp_d['percentage_dict'].iloc[i]) else 0,
          "effectivePositive": (1- s['pos_p']) *100,
          "ineffectivePositive": s['pos_p'] * 100,
          "effectiveNegative": (1- s['neg_p']) *100,
          "ineffectiveNegative": s['neg_p'] *100
        },
        "socialEmotionalTeaching": {
          "rating": int(s['final_emo_score']),
          "allSessionRanking": s['final_emo_all_p'],
          "tutorSessionRanking": s['final_emo_tutor_p'],
          "tutorUtterances": ast.literal_eval(s['t_emo']),
          "studentUtterances": ast.literal_eval(s['s_emo']),
          "time":  ast.literal_eval(s['x_emo']),
          "analysis": s['words']
        }
      }

    }

  }
  
  t = temp_d[temp_d['Tutor ID'] ==  s['Tutor ID']]
  td =  {
    "id": s ['uid'],
    "gender": 'N/A',
    "tutor": s['Tutor ID'],
    "subjects": list(set(t['subject'])),
    "tutoringExperience": 'N/A',
    "sessions": len(t),
    "students": t['student_id'].nunique(),
    "trackingHistory": json.loads(t[['session_uid', 'subject', 'tutoring session occurred date']].rename(columns={'session_uid': 'sessionId', 'subject': 'subject', 'tutoring session occurred date': 'date'}).to_json(orient='records', indent=2)),
    "scoreTrend":{
      "instructionalDelivery": list(t['final_instru_score']),
      "techToolUsage": list(t['final_tech_score']),
      "feedbackQuality": list(t['final_feedback_score']),
      "socialEmotionalTeaching": list(t['final_emo_score']),
      "time":  list(range(len(t)))
    }
  }
  
  tl =   {
    "tutorId": s['Tutor ID'],
    "tutor": s['Tutor ID'],
    "subjects": list(set(t['subject']))
  }
  return dl,sd,td, tl