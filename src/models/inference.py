"""
Inference pipeline v6.1 — Camber-Based Precision Design

KEY FIX: Camber decomposition + scaling
- Decompose CST into thickness + camber components
- Scale camber to directly control lift
- Sensitivity-based Newton iteration for precision
- Create camber variants from CVAE base shapes

Target: <3% error on ALL cases
"""

import torch
import numpy as np
import pickle
import os
import time

from src.models.generator import CVAE
from src.models.forward_model import ForwardModel
from src.geometry.cst import (
    cst_to_coordinates, compute_airfoil_properties, validate_airfoil
)
from src.geometry.export import export_dat, export_csv, export_json


class GeneratedAirfoil:
    """Represents a single generated airfoil design."""

    def __init__(self, cst_upper, cst_lower, predicted_cl, predicted_cd,
                 predicted_cm, properties, x_upper, y_upper, x_lower, y_lower,
                 design_id=0):
        self.cst_upper = cst_upper
        self.cst_lower = cst_lower
        self.predicted_cl = predicted_cl
        self.predicted_cd = predicted_cd
        self.predicted_cm = predicted_cm
        self.properties = properties
        self.x_upper = x_upper
        self.y_upper = y_upper
        self.x_lower = x_lower
        self.y_lower = y_lower
        self.design_id = design_id
        self.thickness = properties.get('max_thickness', 0)

        self.xfoil_cl = None
        self.xfoil_cd = None
        self.xfoil_cm = None
        self.xfoil_verified = False
        self.refined = False
        self.refine_steps = 0

        self.cl_error = None
        self.confidence_score = None
        self.failed_constraints = []

    def export_dat(self, filepath):
        name = f"AirfoilGen_Design_{self.design_id:03d}"
        export_dat(self.x_upper, self.y_upper,
                   self.x_lower, self.y_lower, filepath, name)

    def export_csv(self, filepath):
        metadata = {
            'Max_Thickness': f"{self.thickness:.4f}",
        }
        if self.xfoil_verified:
            metadata['XFOIL_Cl'] = f"{self.xfoil_cl:.4f}"
            metadata['XFOIL_Cd'] = f"{self.xfoil_cd:.6f}"
            metadata['XFOIL_Cm'] = f"{self.xfoil_cm:.4f}"
        name = f"AirfoilGen_Design_{self.design_id:03d}"
        export_csv(self.x_upper, self.y_upper,
                   self.x_lower, self.y_lower, filepath, name, metadata)

    def export_json(self, filepath):
        metadata = {
            'max_thickness': float(self.thickness),
            'cst_upper': self.cst_upper.tolist(),
            'cst_lower': self.cst_lower.tolist(),
            'refined': self.refined,
        }
        if self.xfoil_verified:
            metadata['xfoil_Cl'] = float(self.xfoil_cl)
            metadata['xfoil_Cd'] = float(self.xfoil_cd)
            metadata['xfoil_Cm'] = float(self.xfoil_cm)
        name = f"AirfoilGen_Design_{self.design_id:03d}"
        export_json(self.x_upper, self.y_upper,
                    self.x_lower, self.y_lower, filepath, name, metadata)

    def plot(self, ax=None, show=True, save=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(self.x_upper, self.y_upper, 'b-', linewidth=2, label='Upper')
        ax.plot(self.x_lower, self.y_lower, 'r-', linewidth=2, label='Lower')
        ax.fill_between(self.x_upper, self.y_upper, self.y_lower,
                        alpha=0.1, color='blue')
        ax.set_xlim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        if self.xfoil_verified:
            title = (f'#{self.design_id} | Cl={self.xfoil_cl:.3f} | '
                     f'Cd={self.xfoil_cd:.5f} | Cm={self.xfoil_cm:.4f} | '
                     f't/c={self.thickness:.1%}')
        else:
            title = f'#{self.design_id} | t/c={self.thickness:.1%}'
        ax.set_title(title)
        ax.legend()
        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        return ax

    def __repr__(self):
        if self.xfoil_verified:
            return (f"Airfoil(#{self.design_id}, Cl={self.xfoil_cl:.3f}, "
                    f"Cd={self.xfoil_cd:.5f})")
        return f"Airfoil(#{self.design_id}, unverified)"


class AirfoilGenerator:
    """
    Precision Airfoil Generator with Camber-Based Refinement.

    Pipeline:
    1. CVAE generates base shapes
    2. Create camber variants (scale camber for different Cl)
    3. XFOIL evaluates diverse candidates
    4. Newton-like refinement using camber scaling
    5. Return precision-matched, XFOIL-verified designs
    """

    def __init__(self, checkpoint_dir='checkpoints', device=None):
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        print(f"[GENERATOR] Loading models from {checkpoint_dir}...")

        # Load CVAE
        self.cvae = CVAE(n_cst=16, condition_dim=5, latent_dim=32)
        cvae_path = os.path.join(checkpoint_dir, 'generator_best.pt')
        if os.path.exists(cvae_path):
            ckpt = torch.load(cvae_path, map_location=self.device,
                              weights_only=False)
            self.cvae.load_state_dict(ckpt['model_state_dict'])
            print(f"[GENERATOR] CVAE loaded ✓")
        self.cvae = self.cvae.to(self.device)
        self.cvae.eval()

        # Load Forward Model
        self.forward_model = ForwardModel(input_dim=18)
        self.forward_model_loaded = False
        for fname in ['forwardmodel_best.pt', 'forward_model_best.pt']:
            fwd_path = os.path.join(checkpoint_dir, fname)
            if os.path.exists(fwd_path):
                ckpt = torch.load(fwd_path, map_location=self.device,
                                  weights_only=False)
                self.forward_model.load_state_dict(ckpt['model_state_dict'])
                self.forward_model_loaded = True
                print(f"[GENERATOR] Forward model loaded ✓ ({fname})")
                break
        self.forward_model = self.forward_model.to(self.device)
        self.forward_model.eval()

        # Load scaler
        scaler_path = os.path.join(checkpoint_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"[GENERATOR] Scaler loaded ✓")
        else:
            self.scaler = None

        # Check XFOIL
        self.xfoil_available = False
        try:
            from src.data.xfoil_runner import analyze_airfoil
            self.xfoil_available = True
            print(f"[GENERATOR] XFOIL available ✓")
        except ImportError:
            print(f"[GENERATOR] ⚠ XFOIL not available")

        print(f"[GENERATOR] Ready!")

    # ═══════════════════════════════════════════════
    # XFOIL EVALUATION
    # ═══════════════════════════════════════════════
    def _xfoil_evaluate(self, cst_upper, cst_lower, Re, alpha):
        """Run XFOIL, return (Cl, Cd, Cm) or None."""
        if not self.xfoil_available:
            return None
        from src.data.xfoil_runner import analyze_airfoil
        try:
            _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(
                cst_upper, cst_lower, n_points=100
            )
            results, _, _ = analyze_airfoil(
                x_u, y_u, x_l, y_l,
                reynolds_numbers=[Re], alpha_range=[alpha],
                max_iter=100, timeout=15
            )
            if results:
                c = min(results, key=lambda r: abs(r['alpha'] - alpha))
                return c['Cl'], c['Cd'], c['Cm']
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════
    # CAMBER DECOMPOSITION
    # ═══════════════════════════════════════════════
    @staticmethod
    def _decompose_camber(cst_upper, cst_lower):
        """
        Decompose CST into thickness and camber components.
        
        CST_upper = thickness_part + camber_part
        CST_lower = thickness_part - camber_part
        
        Returns thickness_part, camber_part
        """
        thickness_part = (cst_upper + cst_lower) / 2.0
        camber_part = (cst_upper - cst_lower) / 2.0
        return thickness_part, camber_part

    @staticmethod
    def _compose_from_camber(thickness_part, camber_part, camber_scale=1.0):
        """
        Reconstruct CST from thickness + scaled camber.
        
        camber_scale > 1.0: more camber → more lift
        camber_scale < 1.0: less camber → less lift
        camber_scale = 0.0: symmetric airfoil
        """
        new_upper = thickness_part + camber_scale * camber_part
        new_lower = thickness_part - camber_scale * camber_part
        
        # Clamp to safe CST range
        new_upper = np.clip(new_upper, 0.01, 0.50)
        new_lower = np.clip(new_lower, 0.005, 0.40)
        
        return new_upper, new_lower

    def _validate_cst(self, cst_upper, cst_lower):
        """Check if CST params produce a valid airfoil."""
        try:
            _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(
                cst_upper, cst_lower, n_points=100
            )
            valid, _ = validate_airfoil(x_u, y_u, y_l)
            return valid
        except Exception:
            return False

    # ═══════════════════════════════════════════════
    # CAMBER VARIANTS — Creates diverse Cl shapes
    # ═══════════════════════════════════════════════
    def _create_camber_variants(self, base_shapes, camber_scales=None):
        """
        From each base CVAE shape, create multiple camber variants.
        
        This overcomes CVAE mode collapse by directly varying the 
        camber component while preserving the learned thickness distribution.
        
        Parameters
        ----------
        base_shapes : list of (cst_upper, cst_lower) tuples
        camber_scales : list of float
            Scale factors for camber. >1 = more lift, <1 = less lift
        
        Returns
        -------
        variants : list of (cst_upper, cst_lower) tuples
        """
        if camber_scales is None:
            camber_scales = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9,
                             1.0, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5]
        
        variants = []
        
        for upper, lower in base_shapes:
            thick, camber = self._decompose_camber(upper, lower)
            
            # If camber is too small, add a default
            if np.mean(np.abs(camber)) < 0.005:
                camber = np.array([0.01, 0.02, 0.03, 0.04,
                                   0.04, 0.03, 0.02, 0.01])
            
            for scale in camber_scales:
                new_upper, new_lower = self._compose_from_camber(
                    thick, camber, scale
                )
                
                if self._validate_cst(new_upper, new_lower):
                    variants.append((new_upper.copy(), new_lower.copy()))
        
        return variants

    # ═══════════════════════════════════════════════
    # NEWTON-LIKE CAMBER REFINEMENT
    # ═══════════════════════════════════════════════
    def _refine_camber(self, cst_upper, cst_lower, target_cl, Re, alpha,
                       max_steps=12, cl_tolerance=0.02):
        """
        Precision Cl matching using camber scaling with Newton iteration.
        
        Algorithm:
        1. Evaluate base shape → get Cl_0
        2. Evaluate at +10% camber → get Cl_1
        3. Estimate sensitivity: dCl/df = (Cl_1 - Cl_0) / 0.1
        4. Compute required f: f = f_current + (target - current) / sensitivity
        5. Evaluate, update sensitivity, repeat
        
        This converges in 3-5 XFOIL calls instead of 8+ tiny steps.
        """
        thick, camber = self._decompose_camber(cst_upper, cst_lower)
        
        # Handle near-zero camber
        if np.mean(np.abs(camber)) < 0.005:
            camber = np.array([0.01, 0.02, 0.03, 0.04,
                               0.04, 0.03, 0.02, 0.01])
        
        # Step 1: Evaluate at current scale (f=1.0)
        f_current = 1.0
        result_current = self._xfoil_evaluate(cst_upper, cst_lower, Re, alpha)
        if result_current is None:
            return cst_upper, cst_lower, None, 0, []
        
        cl_current = result_current[0]
        history = [cl_current]
        
        best_f = f_current
        best_error = abs(cl_current - target_cl)
        best_result = result_current
        best_upper = cst_upper.copy()
        best_lower = cst_lower.copy()
        
        # Check if already good enough
        if best_error / max(abs(target_cl), 0.01) < cl_tolerance:
            return cst_upper, cst_lower, result_current, 0, history
        
        # Step 2: Evaluate at perturbed scale to get sensitivity
        f_test = f_current + 0.15  # 15% more camber
        upper_test, lower_test = self._compose_from_camber(
            thick, camber, f_test
        )
        result_test = self._xfoil_evaluate(upper_test, lower_test, Re, alpha)
        
        if result_test is None:
            # Try other direction
            f_test = f_current - 0.15
            upper_test, lower_test = self._compose_from_camber(
                thick, camber, f_test
            )
            result_test = self._xfoil_evaluate(
                upper_test, lower_test, Re, alpha
            )
        
        if result_test is None:
            return best_upper, best_lower, best_result, 0, history
        
        cl_test = result_test[0]
        history.append(cl_test)
        
        # Update best
        if abs(cl_test - target_cl) < best_error:
            best_error = abs(cl_test - target_cl)
            best_f = f_test
            best_result = result_test
            best_upper, best_lower = self._compose_from_camber(
                thick, camber, f_test
            )
        
        # Compute sensitivity
        dCl_df = (cl_test - cl_current) / (f_test - f_current)
        
        if abs(dCl_df) < 0.01:
            return best_upper, best_lower, best_result, 1, history
        
        # Step 3: Newton iterations
        current_f = best_f
        current_cl = best_result[0]
        
        for step in range(max_steps - 2):  # already used 2 evaluations
            # Newton step: f_new = f_current + (target - current) / sensitivity
            cl_gap = target_cl - current_cl
            df = cl_gap / dCl_df
            
            # Limit step size (prevent wild jumps)
            df = np.clip(df, -0.8, 0.8)
            
            new_f = current_f + df
            
            # Keep f in physical range
            new_f = np.clip(new_f, -0.5, 4.0)
            
            # Apply
            new_upper, new_lower = self._compose_from_camber(
                thick, camber, new_f
            )
            
            # Validate
            if not self._validate_cst(new_upper, new_lower):
                df *= 0.5
                new_f = current_f + df
                new_upper, new_lower = self._compose_from_camber(
                    thick, camber, new_f
                )
                if not self._validate_cst(new_upper, new_lower):
                    continue
            
            # XFOIL evaluate
            result = self._xfoil_evaluate(new_upper, new_lower, Re, alpha)
            if result is None:
                df *= 0.3
                continue
            
            new_cl = result[0]
            history.append(new_cl)
            
            # Update sensitivity (secant method)
            if abs(new_f - current_f) > 0.001:
                dCl_df = (new_cl - current_cl) / (new_f - current_f)
                if abs(dCl_df) < 0.01:
                    dCl_df = 0.5 if dCl_df >= 0 else -0.5
            
            current_f = new_f
            current_cl = new_cl
            
            # Track best
            if abs(new_cl - target_cl) < best_error:
                best_error = abs(new_cl - target_cl)
                best_f = new_f
                best_result = result
                best_upper = new_upper.copy()
                best_lower = new_lower.copy()
            
            # Check convergence
            if best_error / max(abs(target_cl), 0.01) < cl_tolerance:
                break
        
        n_steps = len(history) - 1
        return best_upper, best_lower, best_result, n_steps, history

    # ═══════════════════════════════════════════════
    # CVAE CANDIDATE GENERATION
    # ═══════════════════════════════════════════════
    def _generate_cvae_shapes(self, conditions_norm, n_shapes):
        """Generate base shapes from CVAE with temperature scaling."""
        all_cst = []
        conditions_tensor = torch.from_numpy(
            conditions_norm.astype(np.float32)
        ).to(self.device)
        if conditions_tensor.dim() == 1:
            conditions_tensor = conditions_tensor.unsqueeze(0)

        temps = [0.5, 0.8, 1.0, 1.3, 1.8, 2.5]
        per_temp = max(1, n_shapes // len(temps))

        for temp in temps:
            with torch.no_grad():
                cond = conditions_tensor.expand(per_temp, -1)
                z = torch.randn(per_temp, self.cvae.latent_dim,
                                device=self.device) * temp
                cst = self.cvae.decode(z, cond)
                all_cst.append(cst)

        result = torch.cat(all_cst, dim=0)[:n_shapes]

        # Denormalize
        if self.scaler is not None and 'cst_mean' in self.scaler:
            cst_mean = torch.tensor(self.scaler['cst_mean'],
                                     device=self.device, dtype=torch.float32)
            cst_std = torch.tensor(self.scaler['cst_std'],
                                    device=self.device, dtype=torch.float32)
            result = result * cst_std + cst_mean

        # Convert to list of (upper, lower) tuples
        shapes = []
        for i in range(len(result)):
            cst = result[i].cpu().numpy()
            upper = cst[:8]
            lower = cst[8:]
            if self._validate_cst(upper, lower):
                shapes.append((upper.copy(), lower.copy()))

        return shapes

    # ═══════════════════════════════════════════════
    # MAIN GENERATION PIPELINE
    # ═══════════════════════════════════════════════
    def generate(self, Cl, Re, alpha, max_Cd=None, min_thickness=None,
                 max_thickness=None, n_designs=10, n_candidates=100,
                 verify_xfoil=True, n_xfoil_initial=50,
                 n_refine=5, refine_steps=12, cl_tolerance=0.02):
        """
        Generate precision airfoil designs.
        
        Parameters
        ----------
        Cl : float — Target lift coefficient
        Re : float — Reynolds number
        alpha : float — Angle of attack (degrees)
        max_Cd : float — Maximum drag constraint
        min_thickness, max_thickness : float — Thickness constraints
        n_designs : int — Number of designs to return
        n_candidates : int — Base CVAE shapes to generate
        n_xfoil_initial : int — XFOIL evaluations in initial screening
        n_refine : int — Top candidates to refine
        refine_steps : int — Max Newton iterations per candidate
        cl_tolerance : float — Target accuracy (0.02 = 2%)
        """
        start_time = time.time()

        print(f"\n{'='*65}")
        print(f"  AIRFOILGEN v6.1 — Camber-Based Precision Design")
        print(f"{'='*65}")
        print(f"  Target Cl     = {Cl:.3f}")
        print(f"  Re            = {Re:,.0f}")
        print(f"  α             = {alpha:.1f}°")
        print(f"  Cl tolerance  = ±{cl_tolerance*100:.0f}%")

        # ── Conditioning ──
        estimated_thickness = 0.12
        if min_thickness and max_thickness:
            estimated_thickness = (min_thickness + max_thickness) / 2

        cf = 2.0 * 0.074 / (Re ** 0.2)
        cd_form = 0.15 * (estimated_thickness ** 2)
        cd_alpha = 0.02 * (np.radians(abs(alpha)) ** 2)
        estimated_cd = np.clip(cf + cd_form + cd_alpha, 0.005, 0.08)

        conditions_raw = np.array([
            Cl, estimated_cd, np.log10(Re), alpha, estimated_thickness
        ], dtype=np.float32)

        if self.scaler is not None and 'cond_mean' in self.scaler:
            conditions_norm = (
                (conditions_raw - self.scaler['cond_mean']) /
                self.scaler['cond_std']
            )
        else:
            conditions_norm = conditions_raw

        # ═══════════════════════════════════════════
        # PHASE 1: Generate base shapes from CVAE
        # ═══════════════════════════════════════════
        print(f"\n  [PHASE 1] Generating {n_candidates} base shapes...")
        base_shapes = self._generate_cvae_shapes(conditions_norm, n_candidates)
        print(f"    Valid base shapes: {len(base_shapes)}")

        if not base_shapes:
            print(f"  ❌ No valid shapes")
            return []

        # ═══════════════════════════════════════════
        # PHASE 2: Create camber variants
        # ═══════════════════════════════════════════
        # Use a subset of unique base shapes for variants
        unique_base = base_shapes[:min(20, len(base_shapes))]
        
        print(f"  [PHASE 2] Creating camber variants from "
              f"{len(unique_base)} unique bases...")
        
        camber_scales = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9,
                         1.0, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5]
        
        all_variants = self._create_camber_variants(
            unique_base, camber_scales
        )
        
        # Also add original CVAE shapes
        all_variants.extend(base_shapes)
        
        # Remove duplicates (approximately)
        # Filter by thickness constraints
        filtered = []
        for upper, lower in all_variants:
            try:
                _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(
                    upper, lower, n_points=100
                )
                props = compute_airfoil_properties(x_u, y_u, y_l)
                t = props['max_thickness']
                if min_thickness and t < min_thickness:
                    continue
                if max_thickness and t > max_thickness:
                    continue
                filtered.append((upper, lower, props, x_u, y_u, x_l, y_l))
            except Exception:
                continue

        print(f"    Total valid variants: {len(filtered)}")

        if not filtered:
            print(f"  ❌ No valid variants")
            return []

        # ═══════════════════════════════════════════
        # PHASE 3: XFOIL initial screening
        # ═══════════════════════════════════════════
        if verify_xfoil and self.xfoil_available:
            n_eval = min(n_xfoil_initial, len(filtered))
            
            # Pick diverse subset (evenly spaced by thickness)
            filtered.sort(key=lambda x: x[2]['max_thickness'])
            step = max(1, len(filtered) // n_eval)
            to_eval = [filtered[i] for i in range(0, len(filtered), step)][:n_eval]

            print(f"\n  [PHASE 3] XFOIL screening {len(to_eval)} candidates...")
            phase3_start = time.time()

            evaluated = []
            for upper, lower, props, x_u, y_u, x_l, y_l in to_eval:
                result = self._xfoil_evaluate(upper, lower, Re, alpha)
                if result:
                    evaluated.append({
                        'upper': upper, 'lower': lower,
                        'props': props,
                        'x_u': x_u, 'y_u': y_u, 'x_l': x_l, 'y_l': y_l,
                        'xfoil_cl': result[0], 'xfoil_cd': result[1],
                        'xfoil_cm': result[2],
                        'cl_error': abs(result[0] - Cl),
                    })

            phase3_time = time.time() - phase3_start
            print(f"    Converged: {len(evaluated)}/{len(to_eval)} "
                  f"({phase3_time:.1f}s)")

            if not evaluated:
                print(f"  ❌ No XFOIL convergence")
                return []

            # Sort by Cl error
            evaluated.sort(key=lambda d: d['cl_error'])
            best_init = evaluated[0]
            print(f"    Best initial: Cl={best_init['xfoil_cl']:.3f} "
                  f"(err={best_init['cl_error']/abs(Cl)*100:.1f}%)")

            # ═══════════════════════════════════════════
            # PHASE 4: Newton camber refinement
            # ═══════════════════════════════════════════
            n_to_refine = min(n_refine, len(evaluated))
            print(f"\n  [PHASE 4] Refining top {n_to_refine} candidates "
                  f"(Newton method)...")
            phase4_start = time.time()

            refined_designs = []

            for idx in range(n_to_refine):
                cand = evaluated[idx]
                initial_cl = cand['xfoil_cl']
                initial_err = cand['cl_error'] / abs(Cl) * 100

                if initial_err < cl_tolerance * 100:
                    print(f"    #{idx+1}: Already {initial_err:.1f}% — skip ✓")
                    ref_upper = cand['upper']
                    ref_lower = cand['lower']
                    final_result = (cand['xfoil_cl'], cand['xfoil_cd'],
                                    cand['xfoil_cm'])
                    n_steps = 0
                    history = [initial_cl]
                else:
                    ref_upper, ref_lower, final_result, n_steps, history = \
                        self._refine_camber(
                            cand['upper'], cand['lower'],
                            Cl, Re, alpha,
                            max_steps=refine_steps,
                            cl_tolerance=cl_tolerance
                        )

                    if final_result:
                        final_err = abs(final_result[0] - Cl) / abs(Cl) * 100
                        cl_path = '→'.join([f'{h:.3f}' for h in history[:4]])
                        if len(history) > 4:
                            cl_path += f'→...→{history[-1]:.3f}'
                        print(f"    #{idx+1}: Cl [{cl_path}] "
                              f"(err {initial_err:.1f}%→{final_err:.1f}%) "
                              f"[{n_steps} steps]")
                    else:
                        print(f"    #{idx+1}: Refinement failed")
                        continue

                if final_result is None:
                    continue

                # Build design object
                try:
                    _, _, x_u, y_u, x_l, y_l = cst_to_coordinates(
                        ref_upper, ref_lower, n_points=100
                    )
                    props = compute_airfoil_properties(x_u, y_u, y_l)

                    design = GeneratedAirfoil(
                        cst_upper=ref_upper, cst_lower=ref_lower,
                        predicted_cl=final_result[0],
                        predicted_cd=final_result[1],
                        predicted_cm=final_result[2],
                        properties=props,
                        x_upper=x_u, y_upper=y_u,
                        x_lower=x_l, y_lower=y_l,
                        design_id=0
                    )
                    design.xfoil_cl = final_result[0]
                    design.xfoil_cd = final_result[1]
                    design.xfoil_cm = final_result[2]
                    design.xfoil_verified = True
                    design.refined = n_steps > 0
                    design.refine_steps = n_steps
                    design.cl_error = abs(final_result[0] - Cl)

                    if max_Cd and design.xfoil_cd > max_Cd:
                        design.failed_constraints.append('max_cd')

                    refined_designs.append(design)
                except Exception:
                    continue

            phase4_time = time.time() - phase4_start
            print(f"    Refinement complete ({phase4_time:.1f}s)")

            # Also add un-refined but good initial candidates
            for cand in evaluated:
                if cand['cl_error'] / abs(Cl) < 0.10:  # < 10% error
                    already = any(
                        np.allclose(d.cst_upper, cand['upper'], atol=0.001)
                        for d in refined_designs
                    )
                    if not already:
                        try:
                            design = GeneratedAirfoil(
                                cst_upper=cand['upper'],
                                cst_lower=cand['lower'],
                                predicted_cl=cand['xfoil_cl'],
                                predicted_cd=cand['xfoil_cd'],
                                predicted_cm=cand['xfoil_cm'],
                                properties=cand['props'],
                                x_upper=cand['x_u'], y_upper=cand['y_u'],
                                x_lower=cand['x_l'], y_lower=cand['y_l'],
                                design_id=0
                            )
                            design.xfoil_cl = cand['xfoil_cl']
                            design.xfoil_cd = cand['xfoil_cd']
                            design.xfoil_cm = cand['xfoil_cm']
                            design.xfoil_verified = True
                            design.cl_error = cand['cl_error']
                            refined_designs.append(design)
                        except Exception:
                            continue

            # Sort by Cl error
            if max_Cd:
                good_cd = [d for d in refined_designs
                           if 'max_cd' not in d.failed_constraints]
                if good_cd:
                    refined_designs = good_cd

            refined_designs.sort(key=lambda d: d.cl_error)
            final_designs = refined_designs[:n_designs]

        else:
            final_designs = []
            for upper, lower, props, x_u, y_u, x_l, y_l in filtered[:n_designs]:
                design = GeneratedAirfoil(
                    cst_upper=upper, cst_lower=lower,
                    predicted_cl=0, predicted_cd=0, predicted_cm=0,
                    properties=props,
                    x_upper=x_u, y_upper=y_u,
                    x_lower=x_l, y_lower=y_l
                )
                final_designs.append(design)

        # Renumber
        for i, d in enumerate(final_designs):
            d.design_id = i + 1

        # ═══════════════════════════════════════════
        # RESULTS
        # ═══════════════════════════════════════════
        total_time = time.time() - start_time

        print(f"\n{'='*65}")
        print(f"  RESULTS: {len(final_designs)} designs ({total_time:.1f}s)")
        print(f"{'='*65}")

        for d in final_designs:
            if d.xfoil_verified:
                err = d.cl_error / abs(Cl) * 100
                q = ("⭐" if err < 1.0 else "🟢" if err < 2.0 else
                     "🟡" if err < 5.0 else "🟠")
                tag = f" [refined {d.refine_steps}x]" if d.refined else ""
                print(f"  {q} #{d.design_id} | "
                      f"Cl={d.xfoil_cl:+.4f} (err={err:.1f}%) | "
                      f"Cd={d.xfoil_cd:.5f} | "
                      f"Cm={d.xfoil_cm:+.4f} | "
                      f"t/c={d.thickness:.1%}{tag}")

        if final_designs and final_designs[0].xfoil_verified:
            best_err = final_designs[0].cl_error / abs(Cl) * 100
            grade = ("⭐ EXCELLENT" if best_err < 1.0 else
                     "✅ VERY GOOD" if best_err < 2.0 else
                     "✅ GOOD" if best_err < 5.0 else "⚠️ ACCEPTABLE")
            print(f"\n  BEST: {best_err:.1f}% Cl error — {grade}")

        print(f"{'='*65}\n")
        return final_designs

    def _verify_with_xfoil(self, designs, Re, alpha):
        """Legacy compatibility."""
        for d in designs:
            result = self._xfoil_evaluate(d.cst_upper, d.cst_lower, Re, alpha)
            if result:
                d.xfoil_cl, d.xfoil_cd, d.xfoil_cm = result
                d.xfoil_verified = True