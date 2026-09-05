ERP Predictive Workforce Optimization
GitHub Installation and Run Guide

Requirements
Make sure you have the following installed:

Python 3.10 or newer
Git
A web browser such as Chrome, Edge, Firefox, or Safari

Create a Virtual Environment
Run:

python -m venv venv

Activate the Virtual Environment
Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Required Packages

pip install numpy pandas streamlit xgboost scikit-learn ortools

Run the Streamlit Application
Run:

streamlit run app.py

Open the Application
After starting the application, the terminal will display something similar to:

Local URL: http://localhost:8501

Open the displayed URL in your web browser.

Stop the Application
To stop the Streamlit application, return to the terminal and press:

Ctrl + C
