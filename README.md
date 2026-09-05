ERP Predictive Workforce Optimization
GitHub Installation and Run Guide

Requirements
Make sure you have the following installed:

Python 3.10 or newer
Git
A web browser such as Chrome, Edge, Firefox, or Safari

Clone the GitHub Repository
Open Terminal, Command Prompt, or PowerShell and run:

git clone YOUR_GITHUB_REPOSITORY_URL

Then enter the project folder:

cd YOUR_PROJECT_FOLDER

Replace YOUR_GITHUB_REPOSITORY_URL with your actual GitHub repository URL.

Replace YOUR_PROJECT_FOLDER with the name of the downloaded project folder.

Create a Virtual Environment
Run:

python -m venv venv

Activate the Virtual Environment
Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Required Packages
If the project contains a requirements.txt file, run:

pip install -r requirements.txt

If there is no requirements.txt file, run:

pip install numpy pandas streamlit xgboost scikit-learn ortools

Run the Streamlit Application
If the main application file is called app.py, run:

streamlit run app.py

If the main application file has another name, replace app.py with the correct filename.

For example:

streamlit run main.py

Open the Application
After starting the application, the terminal will display something similar to:

Local URL: http://localhost:8501

Open the displayed URL in your web browser.

Stop the Application
To stop the Streamlit application, return to the terminal and press:

Ctrl + C

Quick Installation and Run
If Python and Git are already installed, the complete Windows setup is:

git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

If requirements.txt Does Not Exist
Use:

git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
python -m venv venv
venv\Scripts\activate
pip install numpy pandas streamlit xgboost scikit-learn ortools
streamlit run app.py

Troubleshooting
If "python" is not recognized, install Python 3.10+ and make sure Python is added to PATH.

If "pip" is not recognized, try:

python -m pip install -r requirements.txt

If Streamlit is not recognized, try:

python -m streamlit run app.py

If you receive a missing package error, install the required package with:

pip install PACKAGE_NAME

For example:

pip install ortools

The application should then be available at:

http://localhost:8501
