# Adverse Medical Event Prediction from Phone Calls (USE CASE - 1)
## Team TechIndians

### Demo Video:

A short demo video is available here:

### Problem Statement:

Millions of phone conversations occur daily between patients and nurses regarding medications and health conditions. Often, early signs of adverse medical events go unnoticed.
The objective of this project is to identify and predict potential adverse medical events from recorded phone conversations by leveraging FAERS (FDA Adverse Event Reporting System) data and NLP-based symptom extraction.

### Solution Overview:

Our system analyzes patient-nurse phone calls, extracts medical symptoms using NLP, and maps them to known adverse events from FAERS data.
A severity score is calculated to flag high-risk cases early, enabling timely intervention.

### System Architecture:

- Patient–Nurse Phone Call
- Speech-to-Text Transcription
- Symptom Extraction (NLP)
- FAERS Knowledge Base (2021 Q1–Q4)
- Severity Scoring
- Adverse Event Flagging

### Design & Architecture(With Diagram):

![Screenshot_27-12-2025_104639_www figma com](https://github.com/user-attachments/assets/cf6dabae-3afe-4658-bdbb-af34e337e312)

### Figma Links : 

**LOGO**: https://pen-bloom-84070585.figma.site/

**INFORMATION ARCHITECTURE**: https://cat-floral-16728185.figma.site/

**INTERFACE SCREENS**: https://fence-font-64529008.figma.site/

### Data Source:
- FAERS (FDA Adverse Event Reporting System)
- We worked on Year: 2021 (Q1–Q4)
- **Source**: [https://open.fda.gov/data/faers/](https://open.fda.gov/data/faers/)
- FAERS Tables Used: DEMO : Patient demographics, DRUG : Drug information (Primary Suspect only),REAC : Adverse reactions, OUTC : Outcomes (death, hospitalization, etc.), INDI : Drug indications

### Data Pipeline:
- Automated ingestion of FAERS ASCII files
- Column normalization and cleaning
- Memory-optimized aggregation to avoid data explosion
- Case-level severity scoring using FAERS outcome codes
- Creation of a consolidated dataset for prediction
- (To handle large-scale FAERS data efficiently, reactions and outcomes were aggregated at case level before merging.)

### Quality Assurance (QA):

Testing Approaches Used:
- Manual Test Cases
- Automated Unit Tests

QA Coverage:
- Requirement validation
- Data integrity checks
- Severity score consistency
- Null and edge case handling

(Detailed QA documentation and test cases are available in the /qa folder.)


### Screenshots of UI:

- USER INTERFACE
  

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/35263db2-caf9-4be0-bd3a-8b8646d258c1" />


- USER CLICKS ON BROWSE FILES BUTTON
  
  
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/b12c3fc4-6ab6-48ff-b43e-4f2d6f9cf01b" />


- SELECT THE AUDIO FILE

  
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/bc4ed1c6-d655-4466-b9ca-06e6a417505a" />


- PROCESSING STARTS - TRANSCRIPTION OF AUDIO (audio to text) and TRANSLATION (if audio is in anyother language it is converted to english)


<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/3c693fb2-a003-4dd6-87d9-ebef30015cc7" />


- SYMPTOMS AND DRUGS ARE EXTRACTED


<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/bc109551-baee-43bd-83c7-2d26e29fa51e" />


- The Output is obtained telling the risk of occurrence of adverse event.


Made By:
### Team: Team TechIndians
DCRUST - Batch of 2026
(Veersa Hackathon 2026)
