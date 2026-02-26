"""
╔══════════════════════════════════════════════════════════╗
║  AIRFOIL GENERATION SCRIPT                               ║
║                                                          ║
║  Usage:                                                  ║
║    python scripts/generate_airfoil.py --cl 1.2           ║
║      --re 500000 --alpha 5 --n-designs 10 --verify       ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.inference import AirfoilGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate airfoil designs')

    parser.add_argument('--cl', type=float, required=True,
                        help='Desired lift coefficient')
    parser.add_argument('--re', type=float, required=True,
                        help='Reynolds number')
    parser.add_argument('--alpha', type=float, default=5.0,
                        help='Angle of attack in degrees (default: 5.0)')
    parser.add_argument('--max-cd', type=float, default=None,
                        help='Maximum drag coefficient')
    parser.add_argument('--min-thickness', type=float, default=None,
                        help='Minimum thickness ratio')
    parser.add_argument('--max-thickness', type=float, default=None,
                        help='Maximum thickness ratio')
    parser.add_argument('--n-designs', type=int, default=10,
                        help='Number of designs (default: 10)')
    parser.add_argument('--n-candidates', type=int, default=100,
                        help='Candidates to evaluate (default: 100)')
    parser.add_argument('--output-dir', type=str,
                        default='results/exported_airfoils',
                        help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Model checkpoint directory')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--verify', action='store_true',
                        help='Verify with XFOIL (slower but confirms accuracy)')

    args = parser.parse_args()

    gen = AirfoilGenerator(checkpoint_dir=args.checkpoint_dir)

    designs = gen.generate(
        Cl=args.cl, Re=args.re, alpha=args.alpha,
        max_Cd=args.max_cd,
        min_thickness=args.min_thickness,
        max_thickness=args.max_thickness,
        n_designs=args.n_designs,
        n_candidates=args.n_candidates,
        verify_xfoil=args.verify
    )

    if not designs:
        print("\nNo valid designs. Try relaxing constraints.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nExporting {len(designs)} designs to {args.output_dir}/")

    for design in designs:
        base = f"design_{design.design_id:03d}"
        design.export_dat(os.path.join(args.output_dir, f"{base}.dat"))
        design.export_csv(os.path.join(args.output_dir, f"{base}.csv"))
        design.export_json(os.path.join(args.output_dir, f"{base}.json"))

        if args.plot:
            design.plot(
                save=os.path.join(args.output_dir, f"{base}.png"),
                show=False
            )

    print(f"\n{'='*60}")
    print(f"  GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Designs: {len(designs)}")
    print(f"  Output:  {args.output_dir}")

    if args.verify:
        verified = sum(1 for d in designs if d.xfoil_verified)
        print(f"  XFOIL verified: {verified}/{len(designs)}")

    print(f"\n  Use in XFOIL:")
    print(f"    Load {args.output_dir}/design_001.dat")
    print(f"\n  Use in CAD:")
    print(f"    Import {args.output_dir}/design_001.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()