
# AirfoilGen User Guide

## Introduction
AirfoilGen is an AI-powered tool that generates airfoil shapes based on your performance requirements. Instead of browsing databases for an existing airfoil that *might* work, you can simply ask for "Cl=1.0 at Re=500,000" and get a custom design.

## Installation

### Prerequisites
- Python 3.9+
- [XFOIL](https://web.mit.edu/drela/Public/web/xfoil/) (must be in system PATH or configured)

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify setup:
   ```bash
   python scripts/verify_setup.py
   ```

## Usage

### 1. Interactive Web App (Recommended)
The easiest way to use AirfoilGen is the Streamlit app.

```bash
streamlit run app.py
```

**Features:**
- **Sliders:** Adjust Cl, Reynolds number, and Angle of Attack.
- **Constraints:** Set maximum drag (Cd) and thickness limits.
- **Visualization:** See the airfoil shape and pressure distribution instantly.
- **Download:** Export designs as `.dat` (for XFOIL), `.csv` (for CAD), or `.json`.

### 2. Command Line Interface (CLI)
For batch generation or automation, use the CLI script.

```bash
python scripts/generate_airfoil.py --cl 1.0 --re 500000 --alpha 5.0 --output my_design.dat
```

**Options:**
- `--cl`: Target Lift Coefficient (default: 0.5)
- `--re`: Reynolds Number (default: 500000)
- `--alpha`: Angle of Attack (degrees, default: 5.0)
- `--n_designs`: Number of designs to generate
- `--output`: Output filename (e.g., `wing_design.dat`)
- `--verify`: Run XFOIL verification (slower but accurate)

### 3. Jupyter Notebooks
- `notebooks/02_train_and_generate.ipynb`: Full pipeline (Training + Generation demo).

## Troubleshooting

**"No valid designs found"**
- Try relaxing your constraints (e.g., allow higher drag or a wider thickness range).
- Increase the number of candidates in the generation settings.
- Ensure your requested Cl is reasonable (e.g., Cl=2.0 is very hard for a subsonic airfoil).

**XFOIL Verification Failed**
- Ensure XFOIL is installed correctly.
- Run `python scripts/verify_xfoil.py` to check the connection.
