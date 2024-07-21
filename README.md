# Varsity Tutors & CMU METALS Capstone
Repository of the code for Capston Team Sky-Jem.


## Structure
Data
  - da_schema.csv : Dialogue act classification schema used for MVP1
  - general_rubric.csv : General rubric for session quality assessment for MVP1
  - 500 Sample Supplementary Data (Laste edit_ Mar 21) - mar20 base tutoring session data.csv : Session inforamtion
    
MVP1: Run each in order 
  - Speaker Diarization.ipynb : Generate speaker seperated transcript from audio .wav
  - Entire rubric.ipynb : Overall evaluation for transcript based on rubric
  - DA_prompting.ipynb : Generate dialogue act lables from transcript
  - DA_Visualization.ipynb : Generate insights from da labeled transcript

MVP2: Run this after MVP1 
  - sessionMeasurement_Json .ipynb : Generate Json files for frontend dashboard, https://github.com/kevin666iiiii/Capstone-Web.github.io/tree/main.

MVP3: MVP3 notes

code: Latest python files to process audio.mp3 files to generate analysis result in json format to frontend dashboard.
  - Run in order of 1, 2, 3, 4 

Model: Trained model .pth files
    
Notebook: Random notebooks during exploration and development 

