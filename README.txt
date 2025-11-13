AFTERSHOOT WHITE BALANCE PROJECT - VS CODE SETUP GUIDE

How to Run this Project:

1. Open this folder in VS Code:
   E:\Aftershoot_FullProject

2. Open VS Code Terminal:
   Press: Ctrl + `

3. Create a Virtual Environment:
   python -m venv .venv

4. Activate Virtual Environment:
   PowerShell:
       .\.venv\Scripts\Activate.ps1

5. Install all required libraries:
   pip install -r requirements.txt

6. Run the training + prediction script:
   python train_predict.py --data_dir "dataset"

7. When training finishes, your final submission file will be created here:
   dataset\Validation\submission.csv

Upload the generated submission.csv to the competition portal.
