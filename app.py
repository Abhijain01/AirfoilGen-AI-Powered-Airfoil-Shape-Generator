"""
╔══════════════════════════════════════════════════════════╗
║  AirfoilGen v6.1 — Interactive Airfoil Design Generator  ║
║  Run with: streamlit run app.py                          ║
║                                                          ║
║  FEATURES:                                               ║
║  - Slider + Text input for all parameters                ║
║  - NACA code estimation for each design                  ║
║  - Predicted vs XFOIL comparison table                   ║
║  - High-visibility design cards                          ║
║  - Newton camber refinement                              ║
╚══════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import time
import pandas as pd

# Project setup
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# XFOIL path
xfoil_path = r"C:\Users\abhis\XFOIL6.99"
if os.path.exists(xfoil_path) and xfoil_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] += ";" + xfoil_path

from src.geometry.cst import cst_to_coordinates, compute_airfoil_properties

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="AirfoilGen — AI Airfoil Designer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1E88E5 0%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .version-badge {
        background: linear-gradient(135deg, #43A047 0%, #66BB6A 100%);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .design-card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .design-card-excellent {
        background: #E8F5E9;
        border-left: 5px solid #2E7D32;
        color: #1B5E20;
    }
    .design-card-good {
        background: #E3F2FD;
        border-left: 5px solid #1565C0;
        color: #0D47A1;
    }
    .design-card-ok {
        background: #FFF3E0;
        border-left: 5px solid #E65100;
        color: #BF360C;
    }
    .design-card-poor {
        background: #FFEBEE;
        border-left: 5px solid #C62828;
        color: #B71C1C;
    }
    .design-card-unverified {
        background: #F5F5F5;
        border-left: 5px solid #9E9E9E;
        color: #424242;
    }
    .naca-badge {
        background: #263238;
        color: #4FC3F7;
        padding: 0.15rem 0.6rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 0.9rem;
        letter-spacing: 1px;
    }
    .accuracy-box {
        text-align: center;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .accuracy-excellent {
        background: linear-gradient(135deg, #2E7D32 0%, #43A047 100%);
        color: white;
    }
    .accuracy-good {
        background: linear-gradient(135deg, #1565C0 0%, #1E88E5 100%);
        color: white;
    }
    .accuracy-ok {
        background: linear-gradient(135deg, #E65100 0%, #EF6C00 100%);
        color: white;
    }
    .xfoil-tag {
        background: #E8F5E9;
        border: 1px solid #A5D6A7;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        color: #2E7D32;
        margin: 0.5rem 0;
    }
    .compare-header {
        background: #37474F;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
        font-weight: bold;
        text-align: center;
    }
    .compare-row-good {
        background: #E8F5E9;
        padding: 0.3rem 0.5rem;
        border-radius: 4px;
        margin: 2px 0;
    }
    .compare-row-bad {
        background: #FFEBEE;
        padding: 0.3rem 0.5rem;
        border-radius: 4px;
        margin: 2px 0;
    }
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════

@st.cache_resource
def load_generator(version=5):
    """Load the v6.1 generator model."""
    try:
        import importlib
        import src.models.inference
        importlib.reload(src.models.inference)
        from src.models.inference import AirfoilGenerator

        checkpoint_dir = os.path.join(project_root, 'checkpoints')
        gen = AirfoilGenerator(checkpoint_dir=checkpoint_dir, device='cpu')
        return gen, True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, False


# ═══════════════════════════════════════════════════════
# FORWARD MODEL PREDICTION (for comparison)
# ═══════════════════════════════════════════════════════

def get_forward_model_predictions(generator, designs, Re, alpha):
    """
    Run forward model on each design to get AI predictions.
    Used to compare Predicted vs XFOIL.
    
    Returns dict mapping design_id → {pred_cl, pred_cd, pred_cm}
    """
    predictions = {}
    
    if not generator.forward_model_loaded:
        return predictions
    
    import torch
    
    for design in designs:
        try:
            cst = np.concatenate([design.cst_upper, design.cst_lower])
            forward_input = np.concatenate([
                cst, [alpha], [np.log10(Re)]
            ]).astype(np.float32)
            
            # Normalize
            if generator.scaler is not None and 'input_mean' in generator.scaler:
                forward_input_norm = (
                    (forward_input - generator.scaler['input_mean']) /
                    generator.scaler['input_std']
                )
            else:
                forward_input_norm = forward_input
            
            # Predict
            with torch.no_grad():
                fi_tensor = torch.from_numpy(
                    forward_input_norm
                ).unsqueeze(0).to(generator.device)
                pred = generator.forward_model(fi_tensor).cpu().numpy()[0]
            
            # Denormalize
            if generator.scaler is not None and 'target_mean' in generator.scaler:
                pred = (pred * generator.scaler['target_std'] +
                        generator.scaler['target_mean'])
            
            pred_cl = float(pred[0])
            pred_cd = float(10 ** np.clip(pred[1], -5, 0))
            pred_cm = float(pred[2])
            
            predictions[design.design_id] = {
                'pred_cl': pred_cl,
                'pred_cd': pred_cd,
                'pred_cm': pred_cm,
            }
        except Exception:
            predictions[design.design_id] = {
                'pred_cl': None,
                'pred_cd': None,
                'pred_cm': None,
            }
    
    return predictions


# ═══════════════════════════════════════════════════════
# NACA CODE ESTIMATION
# ═══════════════════════════════════════════════════════

def estimate_naca_code(design):
    """
    Estimate closest NACA 4-digit code from geometry.
    NACA MPXX: M=camber%, P=camber_pos/10, XX=thickness%
    """
    props = design.properties
    max_camber = abs(props.get('max_camber', 0))
    max_camber_loc = props.get('max_camber_loc', 0.3)
    max_thickness = props.get('max_thickness', 0.12)
    
    M = max(0, min(9, int(round(max_camber * 100))))
    P = max(0, min(9, int(round(max_camber_loc * 10))))
    if M == 0:
        P = 0
    XX = max(1, min(40, int(round(max_thickness * 100))))
    
    naca_code = f"{M}{P}{XX:02d}"
    
    if M == 0:
        desc = f"Symmetric, {XX}% thick"
    else:
        desc = f"{M}% camber at {P * 10}% chord, {XX}% thick"
    
    return naca_code, desc


# ═══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def get_quality_grade(cl_error_pct):
    """Return quality emoji, label, CSS class."""
    if cl_error_pct < 1.0:
        return "⭐", "EXCELLENT", "excellent"
    elif cl_error_pct < 2.0:
        return "🟢", "VERY GOOD", "good"
    elif cl_error_pct < 5.0:
        return "🟡", "GOOD", "ok"
    elif cl_error_pct < 10.0:
        return "🟠", "ACCEPTABLE", "ok"
    else:
        return "🔴", "POOR", "poor"


def compute_error_pct(predicted, actual):
    """Compute percentage error safely."""
    if actual is None or predicted is None:
        return None
    denom = max(abs(actual), 0.001)
    return abs(predicted - actual) / denom * 100


def plot_airfoil(designs, target_cl=None):
    """Create publication-quality airfoil plot."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.8, max(len(designs), 1)))

    for i, design in enumerate(designs):
        color = colors[i]
        naca_code, _ = estimate_naca_code(design)

        ax.fill_between(design.x_upper, design.y_upper, design.y_lower,
                        alpha=0.08, color=color)

        if design.xfoil_verified:
            cl_err = abs(design.xfoil_cl - target_cl) / max(
                abs(target_cl), 0.01) * 100
            label = (f"#{design.design_id} [NACA≈{naca_code}]: "
                     f"Cl={design.xfoil_cl:.3f} ({cl_err:.1f}%), "
                     f"Cd={design.xfoil_cd:.5f}")
        else:
            label = (f"#{design.design_id} [NACA≈{naca_code}]: "
                     f"Cl≈{design.predicted_cl:.3f}")

        ax.plot(design.x_upper, design.y_upper, '-', color=color,
                linewidth=2.5, label=label)
        ax.plot(design.x_lower, design.y_lower, '-', color=color,
                linewidth=2.5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('x/c', fontsize=13)
    ax.set_ylabel('y/c', fontsize=13)
    title = 'Generated Airfoil Designs'
    if target_cl is not None:
        title += f' — Target Cl = {target_cl:.3f}'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    return fig


def plot_single_airfoil_detailed(design, target_cl=None):
    """Create detailed plot for a single airfoil."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    naca_code, naca_desc = estimate_naca_code(design)

    ax = axes[0]
    ax.fill_between(design.x_upper, design.y_upper, design.y_lower,
                    alpha=0.15, color='steelblue')
    ax.plot(design.x_upper, design.y_upper, 'b-', linewidth=2.5,
            label='Upper Surface')
    ax.plot(design.x_lower, design.y_lower, 'r-', linewidth=2.5,
            label='Lower Surface')
    camber = (design.y_upper + design.y_lower) / 2
    ax.plot(design.x_upper, camber, 'g--', linewidth=1, alpha=0.5,
            label='Camber Line')

    thickness = design.y_upper - design.y_lower
    max_t_idx = np.argmax(thickness)
    ax.annotate(f't/c = {thickness[max_t_idx]:.1%}',
                xy=(design.x_upper[max_t_idx],
                    (design.y_upper[max_t_idx] + design.y_lower[max_t_idx]) / 2),
                xytext=(0.6, 0.12),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=11, color='green', fontweight='bold')

    ax.set_xlim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('y/c', fontsize=12)
    if design.xfoil_verified:
        title = (f'#{design.design_id} — NACA ≈ {naca_code}\n'
                 f'Cl={design.xfoil_cl:.4f}, Cd={design.xfoil_cd:.5f}, '
                 f'Cm={design.xfoil_cm:.4f}')
    else:
        title = f'#{design.design_id} — NACA ≈ {naca_code}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(design.x_upper, thickness, 'g-', linewidth=2, label='Thickness')
    ax.plot(design.x_upper, camber, 'b-', linewidth=1.5, label='Camber')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=design.x_upper[max_t_idx], color='g', linestyle=':',
               alpha=0.5, label=f'Max t/c at {design.x_upper[max_t_idx]:.2f}')
    ax.set_xlabel('x/c', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Thickness & Camber Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    return fig


def plot_predicted_vs_xfoil(designs, fwd_preds):
    """Create scatter plot: Predicted vs XFOIL for all designs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    labels = ['Cl', 'Cd', 'Cm']
    
    for idx, (ax, label) in enumerate(zip(axes, labels)):
        pred_vals = []
        xfoil_vals = []
        
        for d in designs:
            if not d.xfoil_verified:
                continue
            p = fwd_preds.get(d.design_id)
            if p is None or p.get(f'pred_{label.lower()}') is None:
                continue
            
            pred_vals.append(p[f'pred_{label.lower()}'])
            if label == 'Cl':
                xfoil_vals.append(d.xfoil_cl)
            elif label == 'Cd':
                xfoil_vals.append(d.xfoil_cd)
            else:
                xfoil_vals.append(d.xfoil_cm)
        
        if not pred_vals:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{label}: Predicted vs XFOIL', fontsize=12)
            continue
        
        pred_vals = np.array(pred_vals)
        xfoil_vals = np.array(xfoil_vals)
        
        # Perfect line
        all_vals = np.concatenate([pred_vals, xfoil_vals])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.1
        ax.plot([vmin - margin, vmax + margin],
                [vmin - margin, vmax + margin],
                'k--', linewidth=1, alpha=0.5, label='Perfect match')
        
        # Data points
        ax.scatter(xfoil_vals, pred_vals, c='#1E88E5', s=80, zorder=5,
                   edgecolors='white', linewidth=1.5)
        
        # Labels for each point
        for i, d in enumerate([d for d in designs if d.xfoil_verified]):
            if i < len(pred_vals):
                ax.annotate(f'#{d.design_id}',
                            xy=(xfoil_vals[i], pred_vals[i]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color='#333')
        
        ax.set_xlabel(f'XFOIL {label} (Ground Truth)', fontsize=11)
        ax.set_ylabel(f'Predicted {label} (AI Model)', fontsize=11)
        ax.set_title(f'{label}: Predicted vs XFOIL', fontsize=12,
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def generate_dat_content(design):
    """Generate .dat file content."""
    naca_code, _ = estimate_naca_code(design)
    lines = [f"AirfoilGen_{design.design_id:03d}_NACA{naca_code}"]
    if design.xfoil_verified:
        lines[0] += f"  Cl={design.xfoil_cl:.4f} Cd={design.xfoil_cd:.6f}"
    lines[0] += "\n"
    for i in range(len(design.x_upper) - 1, -1, -1):
        lines.append(f"  {design.x_upper[i]:.6f}  {design.y_upper[i]:.6f}\n")
    for i in range(1, len(design.x_lower)):
        lines.append(f"  {design.x_lower[i]:.6f}  {design.y_lower[i]:.6f}\n")
    return "".join(lines)


def generate_csv_content(design, fwd_pred=None):
    """Generate CSV content with both predicted and XFOIL values."""
    naca_code, naca_desc = estimate_naca_code(design)
    lines = [
        f"# AirfoilGen Design #{design.design_id}\n",
        f"# NACA Equivalent: {naca_code} ({naca_desc})\n",
    ]
    if fwd_pred and fwd_pred.get('pred_cl') is not None:
        lines.append(f"# AI Predicted: Cl={fwd_pred['pred_cl']:.4f}, "
                     f"Cd={fwd_pred['pred_cd']:.6f}, "
                     f"Cm={fwd_pred['pred_cm']:.4f}\n")
    if design.xfoil_verified:
        lines.append(f"# XFOIL Actual:  Cl={design.xfoil_cl:.4f}, "
                     f"Cd={design.xfoil_cd:.6f}, "
                     f"Cm={design.xfoil_cm:.4f}\n")
    lines.append(f"# Thickness: {design.thickness:.4f}\n")
    lines.append(f"surface,x,y\n")
    for i in range(len(design.x_upper)):
        lines.append(f"upper,{design.x_upper[i]:.6f},{design.y_upper[i]:.6f}\n")
    for i in range(len(design.x_lower)):
        lines.append(f"lower,{design.x_lower[i]:.6f},{design.y_lower[i]:.6f}\n")
    return "".join(lines)


def generate_json_content(design, fwd_pred=None):
    """Generate JSON content with both predicted and XFOIL values."""
    naca_code, naca_desc = estimate_naca_code(design)
    data = {
        "name": f"AirfoilGen_Design_{design.design_id:03d}",
        "naca_equivalent": naca_code,
        "naca_description": naca_desc,
        "generator_version": "6.1",
        "max_thickness": round(float(design.thickness), 4),
        "cst_upper": [round(float(v), 6) for v in design.cst_upper],
        "cst_lower": [round(float(v), 6) for v in design.cst_lower],
        "upper_surface": {
            "x": [round(float(v), 6) for v in design.x_upper],
            "y": [round(float(v), 6) for v in design.y_upper],
        },
        "lower_surface": {
            "x": [round(float(v), 6) for v in design.x_lower],
            "y": [round(float(v), 6) for v in design.y_lower],
        }
    }
    if fwd_pred and fwd_pred.get('pred_cl') is not None:
        data["predicted_Cl"] = round(fwd_pred['pred_cl'], 4)
        data["predicted_Cd"] = round(fwd_pred['pred_cd'], 6)
        data["predicted_Cm"] = round(fwd_pred['pred_cm'], 4)
    if design.xfoil_verified:
        data["xfoil_Cl"] = round(float(design.xfoil_cl), 4)
        data["xfoil_Cd"] = round(float(design.xfoil_cd), 6)
        data["xfoil_Cm"] = round(float(design.xfoil_cm), 4)
    return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<div class="main-header">✈️ AirfoilGen</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'AI-Powered Airfoil Shape Generator — '
        'XFOIL-Verified Precision Design '
        '<span class="version-badge">v6.1</span>'
        '</div>',
        unsafe_allow_html=True
    )

    generator, model_loaded = load_generator(version=5)
    if not model_loaded:
        st.error("⚠️ Model not loaded. Ensure checkpoints exist.")
        st.stop()

    # ═══════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════

    st.sidebar.markdown("## 🎯 Design Requirements")
    st.sidebar.markdown("---")

    # Target Cl
    st.sidebar.markdown("**Target Lift Coefficient (Cl)**")
    cl_c1, cl_c2 = st.sidebar.columns([3, 1])
    with cl_c2:
        cl_typed = st.number_input("Cl", -0.5, 2.5, 1.0, 0.01,
                                    format="%.3f", key="cl_t",
                                    label_visibility="collapsed")
    with cl_c1:
        cl_target = st.slider("Cl", -0.5, 2.5, float(cl_typed), 0.01,
                               key="cl_s", label_visibility="collapsed",
                               help="0.2-0.5 cruise, 0.8-1.2 climb, 1.2+ landing")

    if cl_target < 0.3:
        st.sidebar.caption("📝 Low lift — symmetric/reflex")
    elif cl_target < 0.8:
        st.sidebar.caption("📝 Moderate — cruise")
    elif cl_target < 1.3:
        st.sidebar.caption("📝 High — climb/approach")
    else:
        st.sidebar.caption("📝 Very high — thick/cambered")

    st.sidebar.markdown("---")

    # Reynolds
    st.sidebar.markdown("**Reynolds Number (Re)**")
    re_c1, re_c2 = st.sidebar.columns([3, 1])
    with re_c2:
        re_typed = st.number_input("Re", 50000, 10000000, 500000, 50000,
                                    key="re_t", label_visibility="collapsed")
    with re_c1:
        re_value = st.slider("Re", 50000, 10000000, int(re_typed), 50000,
                              key="re_s", label_visibility="collapsed",
                              help="100k=UAV, 500k=RC, 1M+=full scale")

    st.sidebar.markdown("---")

    # Alpha
    st.sidebar.markdown("**Angle of Attack (α°)**")
    a_c1, a_c2 = st.sidebar.columns([3, 1])
    with a_c2:
        alpha_typed = st.number_input("α", -10.0, 20.0, 5.0, 0.1,
                                       format="%.1f", key="a_t",
                                       label_visibility="collapsed")
    with a_c1:
        alpha_value = st.slider("α", -10.0, 20.0, float(alpha_typed), 0.5,
                                 key="a_s", label_visibility="collapsed")

    st.sidebar.markdown("---")

    # Constraints
    st.sidebar.markdown("### 📏 Constraints")
    use_thickness = st.sidebar.checkbox("Thickness constraints", False)
    min_thickness = max_thickness = None
    if use_thickness:
        tc1, tc2 = st.sidebar.columns(2)
        with tc1:
            min_thickness = st.number_input("Min t/c", 0.02, 0.30, 0.06,
                                             0.01, format="%.2f")
        with tc2:
            max_thickness = st.number_input("Max t/c", 0.02, 0.40, 0.20,
                                             0.01, format="%.2f")

    use_max_cd = st.sidebar.checkbox("Max drag constraint", False)
    max_cd = None
    if use_max_cd:
        max_cd = st.sidebar.number_input("Maximum Cd", 0.003, 0.100, 0.035,
                                          0.001, format="%.4f")

    st.sidebar.markdown("---")

    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    n_designs = st.sidebar.slider("Designs to return", 1, 20, 5)

    precision_mode = st.sidebar.radio(
        "Precision Mode",
        ["⚡ Fast (10-15s)", "🎯 Balanced (15-25s)", "⭐ Maximum (25-45s)"],
        index=1
    )

    if precision_mode.startswith("⚡"):
        n_cand, n_xf, n_ref, ref_st, cl_tol = 100, 25, 3, 6, 0.03
    elif precision_mode.startswith("🎯"):
        n_cand, n_xf, n_ref, ref_st, cl_tol = 200, 40, 5, 10, 0.02
    else:
        n_cand, n_xf, n_ref, ref_st, cl_tol = 400, 60, 8, 15, 0.01

    with st.sidebar.expander("🔧 Advanced"):
        n_cand = st.slider("CVAE candidates", 50, 600, n_cand)
        n_xf = st.slider("XFOIL screenings", 10, 100, n_xf)
        n_ref = st.slider("Refine top N", 1, 15, n_ref)
        ref_st = st.slider("Newton steps", 3, 20, ref_st)
        cl_tol_p = st.slider("Cl tolerance %", 1, 10, int(cl_tol * 100))
        cl_tol = cl_tol_p / 100.0

    # ═══════════════════════════════════════════════════
    # MAIN AREA
    # ═══════════════════════════════════════════════════

    st.markdown("---")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Target Cl", f"{cl_target:.3f}")
    with r2:
        st.metric("Reynolds", f"{re_value:,.0f}" if re_value < 1e6
                   else f"{re_value / 1e6:.1f}M")
    with r3:
        st.metric("Alpha", f"{alpha_value:.1f}°")
    with r4:
        st.metric("Precision", precision_mode.split("(")[0].strip())

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        generate_clicked = st.button("🚀 Generate Airfoils",
                                      type="primary",
                                      use_container_width=True)

    # ═══════════════════════════════════════════════════
    # GENERATION
    # ═══════════════════════════════════════════════════

    if generate_clicked:
        progress = st.progress(0, text="Initializing...")
        start_time = time.time()
        progress.progress(10, text="Phase 1: Generating shapes...")

        designs = generator.generate(
            Cl=cl_target, Re=re_value, alpha=alpha_value,
            max_Cd=max_cd,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            n_designs=n_designs,
            n_candidates=n_cand,
            verify_xfoil=True,
            n_xfoil_initial=n_xf,
            n_refine=n_ref,
            refine_steps=ref_st,
            cl_tolerance=cl_tol
        )

        elapsed = time.time() - start_time
        progress.progress(90, text="Computing AI predictions for comparison...")

        # Get forward model predictions for comparison
        fwd_preds = {}
        if designs:
            fwd_preds = get_forward_model_predictions(
                generator, designs, re_value, alpha_value
            )

        progress.progress(100, text="✅ Complete!")

        if not designs:
            st.error("❌ No valid designs. Relax constraints or increase precision.")
            st.stop()

        st.session_state['designs'] = designs
        st.session_state['fwd_preds'] = fwd_preds
        st.session_state['target_cl'] = cl_target
        st.session_state['target_re'] = re_value
        st.session_state['target_alpha'] = alpha_value
        st.session_state['elapsed'] = elapsed
        time.sleep(0.5)
        progress.empty()

    # ═══════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ═══════════════════════════════════════════════════

    if 'designs' in st.session_state:
        designs = st.session_state['designs']
        fwd_preds = st.session_state.get('fwd_preds', {})
        target_cl = st.session_state['target_cl']
        target_re = st.session_state['target_re']
        target_alpha = st.session_state['target_alpha']
        elapsed = st.session_state['elapsed']

        best = designs[0]

        # ─── Accuracy Banner ───
        if best.xfoil_verified:
            best_err = abs(best.xfoil_cl - target_cl) / max(
                abs(target_cl), 0.01) * 100
            emoji, grade_label, grade_class = get_quality_grade(best_err)
            naca_best, _ = estimate_naca_code(best)

            st.markdown(
                f'<div class="accuracy-box accuracy-{grade_class}">'
                f'<span style="font-size:2rem">{emoji}</span><br>'
                f'<strong style="font-size:1.3rem">{grade_label}</strong><br>'
                f'Best Cl error: {best_err:.1f}% — '
                f'{len(designs)} designs in {elapsed:.1f}s<br>'
                f'<span class="naca-badge">NACA ≈ {naca_best}</span><br>'
                f'<small>All results XFOIL-verified</small>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ─── Metrics ───
        st.markdown("### 📊 Summary")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            if best.xfoil_verified:
                st.metric("Best Cl", f"{best.xfoil_cl:.4f}",
                          delta=f"{best.xfoil_cl - target_cl:+.4f}")
        with m2:
            if best.xfoil_verified:
                st.metric("Best Cd", f"{best.xfoil_cd:.5f}")
        with m3:
            if best.xfoil_verified:
                st.metric("L/D", f"{best.xfoil_cl / max(best.xfoil_cd, 1e-6):.1f}")
        with m4:
            st.metric("t/c", f"{best.thickness:.1%}")
        with m5:
            naca_code, _ = estimate_naca_code(best)
            st.metric("NACA ≈", naca_code)
        with m6:
            n_ref = sum(1 for d in designs
                        if hasattr(d, 'refined') and d.refined)
            st.metric("Refined", f"{n_ref}/{len(designs)}")

        # ─── Shapes ───
        st.markdown("### ✈️ Airfoil Shapes")
        fig = plot_airfoil(designs, target_cl)
        st.pyplot(fig)
        plt.close()

        # ─── Design Cards ───
        st.markdown("### 🏆 Design Rankings")

        for design in designs:
            naca_code, _ = estimate_naca_code(design)

            if design.xfoil_verified:
                cl_err = abs(design.xfoil_cl - target_cl) / max(
                    abs(target_cl), 0.01) * 100
                emoji, _, _ = get_quality_grade(cl_err)
                ld = design.xfoil_cl / max(design.xfoil_cd, 1e-6)

                card_class = ("design-card-excellent" if cl_err < 2 else
                              "design-card-good" if cl_err < 5 else
                              "design-card-ok" if cl_err < 10 else
                              "design-card-poor")

                refined_tag = ""
                if hasattr(design, 'refined') and design.refined:
                    refined_tag = (f'&nbsp;|&nbsp; 🔧 '
                                   f'{design.refine_steps} Newton steps')

                st.markdown(
                    f'<div class="design-card {card_class}">'
                    f'<strong>{emoji} Design #{design.design_id}</strong>'
                    f'&nbsp;&nbsp;'
                    f'<span class="naca-badge">NACA ≈ {naca_code}</span>'
                    f'<br>'
                    f'<strong>Cl = {design.xfoil_cl:+.4f}</strong>'
                    f' (target {target_cl:+.3f}, '
                    f'<strong>{cl_err:.1f}%</strong>)'
                    f'&nbsp;|&nbsp; Cd = {design.xfoil_cd:.5f}'
                    f'&nbsp;|&nbsp; Cm = {design.xfoil_cm:+.4f}'
                    f'&nbsp;|&nbsp; L/D = {ld:.1f}'
                    f'&nbsp;|&nbsp; t/c = {design.thickness:.1%}'
                    f'{refined_tag}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="design-card design-card-unverified">'
                    f'⚠️ #{design.design_id} '
                    f'<span class="naca-badge">NACA ≈ {naca_code}</span>'
                    f' — Cl≈{design.predicted_cl:.3f} (unverified)'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ═══════════════════════════════════════════════
        # PREDICTED vs XFOIL COMPARISON
        # ═══════════════════════════════════════════════
        st.markdown("### 🔬 AI Predicted vs XFOIL Comparison")
        st.markdown("*How accurate is the AI forward model compared to "
                     "XFOIL ground truth?*")

        # Scatter plots
        if fwd_preds:
            fig_compare = plot_predicted_vs_xfoil(designs, fwd_preds)
            st.pyplot(fig_compare)
            plt.close()

        # Comparison table
        comp_rows = []
        for d in designs:
            if not d.xfoil_verified:
                continue
            p = fwd_preds.get(d.design_id, {})
            if p.get('pred_cl') is None:
                continue

            naca, _ = estimate_naca_code(d)
            cl_err = compute_error_pct(p['pred_cl'], d.xfoil_cl)
            cd_err = compute_error_pct(p['pred_cd'], d.xfoil_cd)
            cm_err = compute_error_pct(p['pred_cm'], d.xfoil_cm)

            comp_rows.append({
                'Design': f"#{d.design_id}",
                'NACA': naca,
                'Pred Cl': f"{p['pred_cl']:.4f}",
                'XFOIL Cl': f"{d.xfoil_cl:.4f}",
                'Cl Err%': f"{cl_err:.1f}%" if cl_err else "—",
                'Pred Cd': f"{p['pred_cd']:.5f}",
                'XFOIL Cd': f"{d.xfoil_cd:.5f}",
                'Cd Err%': f"{cd_err:.1f}%" if cd_err else "—",
                'Pred Cm': f"{p['pred_cm']:.4f}",
                'XFOIL Cm': f"{d.xfoil_cm:.4f}",
                'Cm Err%': f"{cm_err:.1f}%" if cm_err else "—",
            })

        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True,
                         use_container_width=True)

            # Summary stats
            cl_errs = [compute_error_pct(
                fwd_preds.get(d.design_id, {}).get('pred_cl'),
                d.xfoil_cl
            ) for d in designs if d.xfoil_verified and
                fwd_preds.get(d.design_id, {}).get('pred_cl') is not None]
            cd_errs = [compute_error_pct(
                fwd_preds.get(d.design_id, {}).get('pred_cd'),
                d.xfoil_cd
            ) for d in designs if d.xfoil_verified and
                fwd_preds.get(d.design_id, {}).get('pred_cd') is not None]

            if cl_errs and cd_errs:
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Mean Cl Error", f"{np.mean(cl_errs):.1f}%")
                with s2:
                    st.metric("Mean Cd Error", f"{np.mean(cd_errs):.1f}%")
                with s3:
                    st.metric("Max Cl Error", f"{np.max(cl_errs):.1f}%")
                with s4:
                    st.metric("Max Cd Error", f"{np.max(cd_errs):.1f}%")

                st.info(
                    "💡 **Why the difference?** The AI forward model was "
                    "trained on real airfoil data (R²>0.999), but the CVAE "
                    "generates shapes slightly outside the training distribution. "
                    "XFOIL provides ground-truth physics simulation for "
                    "every design."
                )

        # ─── Detailed View ───
        st.markdown("### 🔍 Detailed Design View")

        design_options = []
        for d in designs:
            naca, _ = estimate_naca_code(d)
            if d.xfoil_verified:
                cl_err = abs(d.xfoil_cl - target_cl) / max(
                    abs(target_cl), 0.01) * 100
                emoji, _, _ = get_quality_grade(cl_err)
                label = (f"{emoji} #{d.design_id} [NACA≈{naca}] — "
                         f"Cl={d.xfoil_cl:.4f} ({cl_err:.1f}%)")
            else:
                label = f"#{d.design_id} [NACA≈{naca}]"
            design_options.append(label)

        selected_label = st.selectbox("Select design:", design_options)
        selected_idx = design_options.index(selected_label)
        selected = designs[selected_idx]
        sel_naca, sel_naca_desc = estimate_naca_code(selected)
        sel_pred = fwd_preds.get(selected.design_id, {})

        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig2 = plot_single_airfoil_detailed(selected, target_cl)
            st.pyplot(fig2)
            plt.close()

        with col_right:
            # NACA code box
            st.markdown(
                f'<div style="text-align:center; padding:0.8rem; '
                f'background:#263238; border-radius:10px; margin-bottom:1rem;">'
                f'<span style="color:#90CAF9; font-size:0.8rem;">'
                f'NACA Equivalent</span><br>'
                f'<span style="color:#4FC3F7; font-size:2rem; '
                f'font-family:monospace; font-weight:bold; '
                f'letter-spacing:3px;">{sel_naca}</span><br>'
                f'<span style="color:#B0BEC5; font-size:0.75rem;">'
                f'{sel_naca_desc}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ── PREDICTED vs XFOIL TABLE (per design) ──
            if selected.xfoil_verified and sel_pred.get('pred_cl') is not None:
                st.markdown("#### 🔬 Predicted vs XFOIL")

                cl_e = compute_error_pct(sel_pred['pred_cl'], selected.xfoil_cl)
                cd_e = compute_error_pct(sel_pred['pred_cd'], selected.xfoil_cd)
                cm_e = compute_error_pct(sel_pred['pred_cm'], selected.xfoil_cm)

                pvx_data = {
                    "Metric": ["Cl", "Cd", "Cm", "L/D"],
                    "🤖 AI Predicted": [
                        f"{sel_pred['pred_cl']:.4f}",
                        f"{sel_pred['pred_cd']:.6f}",
                        f"{sel_pred['pred_cm']:.4f}",
                        f"{sel_pred['pred_cl'] / max(sel_pred['pred_cd'], 1e-6):.1f}",
                    ],
                    "🔬 XFOIL Actual": [
                        f"{selected.xfoil_cl:.4f}",
                        f"{selected.xfoil_cd:.6f}",
                        f"{selected.xfoil_cm:.4f}",
                        f"{selected.xfoil_cl / max(selected.xfoil_cd, 1e-6):.1f}",
                    ],
                    "Error %": [
                        f"{cl_e:.1f}%" if cl_e else "—",
                        f"{cd_e:.1f}%" if cd_e else "—",
                        f"{cm_e:.1f}%" if cm_e else "—",
                        "—",
                    ],
                }
                st.dataframe(pd.DataFrame(pvx_data), hide_index=True,
                             use_container_width=True)

                st.markdown(
                    '<div class="xfoil-tag">'
                    '✓ XFOIL values are ground truth (physics simulation)'
                    '</div>',
                    unsafe_allow_html=True
                )

            # ── Geometry properties ──
            st.markdown("#### 📋 Geometry")
            geo_data = {
                "Property": [
                    "📏 Thickness (t/c)",
                    "📍 Max t/c Location",
                    "〰️ Max Camber",
                    "⭕ LE Radius",
                    "🏷️ NACA Code",
                    "🔧 Refined",
                ],
                "Value": [
                    f"{selected.thickness:.4f} ({selected.thickness:.1%})",
                    f"{selected.properties.get('max_thickness_loc', 0):.3f}",
                    f"{selected.properties.get('max_camber', 0):.4f}",
                    f"{selected.properties.get('le_radius', 0):.5f}",
                    f"NACA {sel_naca}",
                    f"{'Yes (' + str(selected.refine_steps) + ' steps)' if hasattr(selected, 'refined') and selected.refined else 'No'}",
                ]
            }
            st.dataframe(pd.DataFrame(geo_data), hide_index=True,
                         use_container_width=True)

            # CST
            st.markdown("#### 🧬 CST Parameters")
            st.dataframe(pd.DataFrame({
                "i": [f"w{i}" for i in range(8)],
                "Upper": [f"{v:.6f}" for v in selected.cst_upper],
                "Lower": [f"{v:.6f}" for v in selected.cst_lower],
            }), hide_index=True, use_container_width=True)

        # ─── Coordinates ───
        with st.expander("📐 Coordinates"):
            st.dataframe(pd.concat([
                pd.DataFrame({'Surface': 'Upper',
                              'x': selected.x_upper,
                              'y': selected.y_upper}),
                pd.DataFrame({'Surface': 'Lower',
                              'x': selected.x_lower,
                              'y': selected.y_lower}),
            ], ignore_index=True), use_container_width=True, height=300)

        # ─── Downloads ───
        st.markdown("### 📥 Export")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("📄 .dat (XFOIL)",
                               generate_dat_content(selected),
                               f"NACA{sel_naca}_{selected.design_id}.dat",
                               "text/plain", use_container_width=True)
        with d2:
            st.download_button("📊 .csv (CAD)",
                               generate_csv_content(selected, sel_pred),
                               f"NACA{sel_naca}_{selected.design_id}.csv",
                               "text/csv", use_container_width=True)
        with d3:
            st.download_button("🔧 .json (API)",
                               generate_json_content(selected, sel_pred),
                               f"NACA{sel_naca}_{selected.design_id}.json",
                               "application/json", use_container_width=True)
        with d4:
            all_dat = "\n".join(generate_dat_content(d) for d in designs)
            st.download_button("📦 ALL (.dat)", all_dat,
                               "all_designs.dat", "text/plain",
                               use_container_width=True)

        # ═══════════════════════════════════════════════
        # FULL COMPARISON TABLE
        # ═══════════════════════════════════════════════
        st.markdown("### 📊 Full Design Comparison")

        rows = []
        for d in designs:
            naca, _ = estimate_naca_code(d)
            p = fwd_preds.get(d.design_id, {})

            if d.xfoil_verified:
                cl_err_target = abs(d.xfoil_cl - target_cl) / max(
                    abs(target_cl), 0.01) * 100
                emoji, _, _ = get_quality_grade(cl_err_target)

                row = {
                    '': emoji,
                    'Design': f"#{d.design_id}",
                    'NACA': naca,
                    't/c': f"{d.thickness:.1%}",
                }

                # Predicted values
                if p.get('pred_cl') is not None:
                    row['Pred Cl'] = f"{p['pred_cl']:.4f}"
                    row['Pred Cd'] = f"{p['pred_cd']:.5f}"
                    row['Pred Cm'] = f"{p['pred_cm']:.4f}"
                else:
                    row['Pred Cl'] = "—"
                    row['Pred Cd'] = "—"
                    row['Pred Cm'] = "—"

                # XFOIL values
                row['XFOIL Cl'] = f"{d.xfoil_cl:+.4f}"
                row['XFOIL Cd'] = f"{d.xfoil_cd:.5f}"
                row['XFOIL Cm'] = f"{d.xfoil_cm:+.4f}"
                row['L/D'] = f"{d.xfoil_cl / max(d.xfoil_cd, 1e-6):.1f}"
                row['Target Err'] = f"{cl_err_target:.1f}%"

                # Prediction errors
                if p.get('pred_cl') is not None:
                    row['Cl Err'] = f"{compute_error_pct(p['pred_cl'], d.xfoil_cl):.1f}%"
                    row['Cd Err'] = f"{compute_error_pct(p['pred_cd'], d.xfoil_cd):.0f}%"
                else:
                    row['Cl Err'] = "—"
                    row['Cd Err'] = "—"

                row['Refined'] = "✓" if hasattr(d, 'refined') and d.refined else "—"
            else:
                row = {
                    '': '⚠️', 'Design': f"#{d.design_id}", 'NACA': naca,
                    't/c': f"{d.thickness:.1%}",
                    'Pred Cl': "—", 'Pred Cd': "—", 'Pred Cm': "—",
                    'XFOIL Cl': "—", 'XFOIL Cd': "—", 'XFOIL Cm': "—",
                    'L/D': "—", 'Target Err': "—",
                    'Cl Err': "—", 'Cd Err': "—", 'Refined': "—",
                }

            rows.append(row)

        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)

    # ─── Footer ───
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.85rem;'>
        <strong>AirfoilGen v6.1</strong> — 
        Camber-Based Precision Airfoil Design<br>
        CVAE + XFOIL Verification + Newton Camber Refinement<br>
        Accuracy: <strong>&lt;1% Cl error</strong> | 
        Shows both AI predictions and XFOIL ground truth<br>
        Built with PyTorch + XFOIL + Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()