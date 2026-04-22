# NutriCode Uganda — Prototype

An AI-powered nutritional intelligence system for regionally-adapted phytate
reduction in Uganda's staple foods.

Built for the **Future Makers Hackathon 2026** (National Science Week, Uganda Ministry of Science, Technology and Innovation).

---

## What this prototype demonstrates

1. **AI Recommender** — ranks candidate enzyme/microbial formulations by predicted phytate reduction % for any selected Ugandan district and staple crop.
2. **Uganda Map View** — districts coloured by Global Acute Malnutrition (GAM), anemia rate, or calculated priority score, using IPC Acute Malnutrition Snapshot data (April 2025–February 2026).
3. **SHAP Explainability** — per-prediction feature attribution showing exactly why the model makes each recommendation.
4. **Data & Method** — full transparency on training data sources, model performance metrics, and Uganda-specific datasets.

The prototype is the **AI layer** of a larger research programme. Laboratory validation at Makerere University's Department of Food Technology and Nutrition is the next phase.

---

## Local setup (for development)

```bash
# Clone the repo
git clone <your-repo-url>
cd prototype

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deployment to Streamlit Community Cloud (FREE, recommended)

**Time required: ~10 minutes.**

### Step 1 — Push this folder to GitHub

1. Go to https://github.com/new and create a new **public** repository named `nutricode`
2. On your machine, inside the `prototype/` folder:
   ```bash
   git init
   git add .
   git commit -m "NutriCode Uganda prototype"
   git branch -M main
   git remote add origin https://github.com/<your-username>/nutricode-uganda.git
   git push -u origin main
   ```

### Step 2 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub accoun
3. Click **"New app"**
4. Select your `nutricode-uganda` repository
5. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** `nutricode` (or your preferred subdomain)
6. Click **"Deploy"**

Streamlit Community Cloud will:
- Clone your repo
- Install `requirements.txt`
- Launch your app at `https://nutricode.streamlit.app` (or your chosen subdomain)

The first build takes ~3–5 minutes. After that, every push to `main` redeploys automatically.

### Step 3 — Test it
Open your live URL. Confirm:
- The district dropdown populates (should show Gulu, Lira, etc.)
- The recommender returns 5 ranked strains
- The Uganda map renders with data points
- The SHAP tab shows a feature importance chart

---

## File structure

```
prototype/
├── app.py                        # Main Streamlit application
├── model.py                      # ML models + SHAP + prediction logic
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .streamlit/
│   └── config.toml               # Theme configuration
└── data/
    ├── phytase_performance.csv   # Training data (50 literature samples)
    └── uganda_districts.csv      # 42 Ugandan districts with IPC data
```

---

## Training data sources

**Phytase performance data** (50 samples) is drawn from published research:
- Greiner et al. — fungal phytases, pH optimization
- Fredrikson et al. — Bacillus subtilis phytate reduction
- Hellström et al. — Pichia kudriavzevii rapid degradation
- Songré-Ouattara et al. — Lactobacillus plantarum in African millet
- Nout — maize fermentation studies
- Deleu et al. — microbial consortia for cereal applications

**Uganda district data** combines:
- IPC Acute Food Insecurity and Malnutrition Snapshot — Uganda, April 2025 – February 2026
- Uganda Bureau of Statistics (UBOS) population figures
- UDHS 2016 anemia and stunting baselines
- UBOS / NIPN Nutrition Data Landscape Report (2020)

---

## Models

- **Random Forest Regressor** (200 trees, max depth 10)
- **XGBoost Regressor** (200 rounds, max depth 5, learning rate 0.08)
- **Ensemble** averages both predictions to reduce variance
- **SHAP TreeExplainer** provides per-prediction feature attribution

Performance metrics are displayed in the "Data & Method" tab of the app.

---

## Important to note

This prototype demonstrates **predictive modelling from published literature**. It is not a validated clinical intervention. The next step is:

1. **Phase 1 — Laboratory validation at Makerere University:** test top-ranked formulations on real Ugandan grain samples under realistic cooking conditions.
2. **Phase 2 — Community pilot in Gulu and Lira:** deploy validated formulations through district health worker networks.
3. **Phase 3 — Regional scale:** expand to full Northern Uganda with region-specific formula variants.
4. **Phase 4 — National system:** integrate with MoH CHW network and East African Community licensing pathway.

The hackathon prototype is Phase 0. Everything after requires the partnership, funding, and validation unlocked by winning.

---

## Team

**NutriCode Uganda** is built by;

- **Team Lead** — AI and Software (Patience Sitati)
- **Software/AI Engineer** (Immaculate Kaitesi)
- **Food Nutritionist** (Amucu Esther)
- **Data Scientist** (Joel Mwaka)

---

## License

This prototype is submitted to the Future Makers Hackathon 2026 as intellectual property of NutriCode Uganda. Reuse of training data requires attribution to the original literature sources cited above.
