
# ✈️ AirfoilGen — AI-Powered Airfoil Shape Generator

> **Given desired aerodynamic performance → Generate airfoil coordinates**

AirfoilGen is a deep learning model that designs airfoils to meet specific lift, drag, and moment requirements. It uses a **CVAE (Conditional Variational Autoencoder)** to generate shapes and a **Neural Network Forward Model** to predict their performance instantly.

## 🚀 Features
- **Inverse Design:** Input your target Cl, Re, and Alpha — get an airfoil.
- **Physics-Informed:** Validated against XFOIL physics simulations.
- **Interactive UI:** Streamlit app for real-time design exploration.
- **Export Ready:** Download as `.dat` (XFOIL), `.csv` (CAD), or `.json`.

## 📦 quickstart

1. **Install:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App:**
   ```bash
   streamlit run app.py
   ```

3. **Generate:**
   - Set your target Lift (Cl)
   - Adjust Reynolds number
   - Value your design!

## 📂 Project Structure
- `src/models`: Neural network definitions (CVAE, Forward Model)
- `src/geometry`: CST parametrization and export logic 
- `scripts/`: Training and utility scripts
- `notebooks/`: Jupyter notebooks for experiments
- `checkpoints/`: Saved model weights

## 🛠️ CLI Usage
Generate an airfoil from the terminal:
```bash
python scripts/generate_airfoil.py --cl 0.8 --re 200000 --output custom_foil.dat
```

## 📝 License
MIT License 

## activate venv

.\.venv39\Scripts\Activate.ps1