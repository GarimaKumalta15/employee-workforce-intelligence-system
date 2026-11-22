"""One-command launcher:  python run.py

First run generates the dataset and trains the models (~30 s), then starts the web app.
"""
from app import create_app

if __name__ == "__main__":
    application = create_app()
    print("\n  Employee & Workforce Intelligence System running at  http://127.0.0.1:5000\n")
    application.run(host="127.0.0.1", port=5000, debug=False)
