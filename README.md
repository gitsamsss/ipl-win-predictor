# IPL Win Predictor

A machine learning project that predicts the **real-time win probability** of a team during an IPL (Indian Premier League) cricket match, based on the current match situation — score, wickets in hand, overs completed, and target.

## About

During a run chase in T20 cricket, the outcome depends heavily on the chase dynamics: how many runs are needed, how many balls are left, how many wickets remain, and how the required run rate compares to the current run rate. This project turns that judgment call into a data-driven prediction using historical IPL data.

The model is a **Logistic Regression classifier** trained on ball-by-ball IPL data, wrapped in a scikit-learn `Pipeline` with one-hot encoding for categorical features (teams, city). Given a live match situation, it outputs the win probability for both the batting and bowling teams.

## Files

| File | Description |
|---|---|
| `ipl_win_predictor_jupyter.ipynb` | Jupyter notebook — loads the data, engineers features, trains the model, and provides an interactive `ipywidgets` predictor UI |
| `matches.csv` | Match-level IPL data (teams, city, winner, toss, target score) |
| `deliveries.csv` | Ball-by-ball delivery data (runs, wickets, overs) for every match |

## How It Works

1. **Data prep** — merges match-level and ball-by-ball data, keeping only second-innings (run-chase) data for 8 major IPL franchises
2. **Feature engineering** — derives:
   - `runs_left` — runs still needed
   - `balls_left` — balls remaining
   - `wickets` — wickets in hand
   - `crr` — current run rate
   - `rrr` — required run rate
3. **Preprocessing** — one-hot encodes `batting_team`, `bowling_team`, and `city`
4. **Model** — Logistic Regression, trained via an sklearn `Pipeline`
5. **Prediction** — `.predict_proba()` returns a win % for each team

## Running It

1. Clone this repo:
   ```bash
   git clone https://github.com/gitsamsss/ipl-win-predictor.git
   cd ipl-win-predictor
   ```
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `ipl_win_predictor_jupyter.ipynb` and run all cells (**Cell → Run All**).
   - The notebook installs its own dependencies (`pandas`, `scikit-learn`, `ipywidgets`) in the first cell
   - It trains the model directly from `matches.csv` and `deliveries.csv`, so no pre-trained model file is needed
4. Use the dropdowns and inputs at the bottom of the notebook (batting team, bowling team, city, target, score, wickets, overs) and click **Predict Probability**.

## Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn (Logistic Regression, ColumnTransformer, OneHotEncoder)
- ipywidgets (interactive UI inside Jupyter)

## Dataset Source

Historical IPL ball-by-ball data (matches and deliveries), originally sourced from Kaggle's IPL dataset.
