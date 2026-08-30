# Deployment Guide

## Running Locally

1. Unzip the project and open a terminal inside the `content-moderation-project` folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   python app/app.py
   ```
4. Open http://127.0.0.1:5000 in your browser — type text, click "Check Text."

You do NOT need to retrain anything to run the app. The trained model and
vectorizer (`models/logistic_regression_model.joblib`,
`models/tfidf_vectorizer.joblib`) are already included in this project.

Only re-run the training pipeline if you want to retrain from scratch:
```
python run_preprocessing.py   # needs data/raw/train.csv (see README.md)
python run_features.py
python train.py
python evaluate.py
```

---

## Deploying for Free — Option 1: Render.com (recommended)

Render's free tier is the simplest option for a Flask + scikit-learn app like
this one. The project already includes a `Procfile` and `gunicorn` in
`requirements.txt`, so no extra configuration is needed.

### Steps

1. **Push the project to GitHub** (free account, new repo):
   ```
   git init
   git add .
   git commit -m "Content moderation prototype"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```
   Note: `data/raw/train.csv` does NOT need to be in the repo — the deployed
   app only needs the already-trained files in `models/`, not the raw dataset.

2. Go to **render.com**, sign up free, click **New → Web Service**, and
   connect your GitHub repository.

3. Render should auto-detect Python. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --chdir app app:app --bind 0.0.0.0:$PORT`
   - **Instance Type:** Free

4. Click **Deploy**. Render will build the app and give you a public URL
   like `your-app.onrender.com` after a few minutes.

### Good to know

- The free tier spins the app down after 15 minutes of no traffic. The
  first request after idling takes roughly 30–50 seconds to "wake up."
  This is fine for a project demo or defense, not for production traffic.
- If the build fails on the gunicorn start command, double check the
  `Procfile` in the project root reads exactly:
  ```
  web: gunicorn --chdir app app:app --bind 0.0.0.0:$PORT
  ```

---

## Deploying for Free — Option 2: PythonAnywhere (no Git required)

If GitHub feels like extra overhead, PythonAnywhere's free tier lets you
upload files directly through a web-based file manager and configure a
Flask app through their dashboard — no Procfile or gunicorn setup needed,
they handle that for you.

### Steps

1. Sign up free at **pythonanywhere.com**.
2. Go to the **Files** tab and upload the project folder (or use their
   built-in "Bash console" to `git clone` if you did push to GitHub).
3. Go to the **Web** tab → **Add a new web app** → choose **Flask** →
   select a Python version matching what you used locally.
4. Point the WSGI configuration file to your `app/app.py` and its `app`
   Flask instance (PythonAnywhere's setup wizard walks you through the
   exact file path and variable name to enter).
5. Install dependencies via their **Bash console**:
   ```
   pip install --user -r requirements.txt
   ```
6. Click **Reload** on the Web tab. Your app is live at
   `yourusername.pythonanywhere.com`.

### Good to know

- The free tier is always-on (no spin-down like Render), but has limited
  CPU seconds per day — more than enough for demo/defense purposes.
- Outbound internet access is restricted on the free tier, but this app
  doesn't need any external network calls at runtime, so that's not a
  concern here.

---

## Which one should you actually use?

- **Render** if you're comfortable with (or want to learn) Git — cleaner
  workflow, redeploys automatically on every `git push`.
- **PythonAnywhere** if you want the fastest path with zero command-line
  deployment steps.

Either is genuinely free and sufficient for a university project defense.
Neither is intended for real production traffic — that's expected and fine
for an academic prototype.
