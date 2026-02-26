"""
╔══════════════════════════════════════════════════════════╗
║  DATA GENERATION SCRIPT                                  ║
║                                                          ║
║  Run this to generate ALL training data.                 ║
║  This is Phase 1, Day 3-10 of the project.               ║
║                                                          ║
║  Usage: python scripts/generate_data.py                  ║
║                                                          ║
║  Expected runtime: 30 min to 2 hours                     ║
║  (depending on number of airfoils and XFOIL speed)       ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.helpers import set_seed
from src.utils.logger import setup_logger
from src.data.preprocessing import run_full_pipeline


def main():
    # Load configuration
    config = Config("config.yaml")
    
    # Set up logging
    logger = setup_logger("data_pipeline", log_dir="logs")
    
    # Set random seed
    set_seed(config.project.random_seed)
    
    logger.info("Starting data generation pipeline...")
    logger.info(f"Random seed: {config.project.random_seed}")
    
    try:
        # Run the full pipeline
        data = run_full_pipeline(
            config=config,
            output_dir=config.paths.processed_data
        )
        
        logger.info("Data pipeline completed successfully!")
        logger.info(f"Total data points: {len(data['cl'])}")
        logger.info(f"Train: {data['train_mask'].sum()}")
        logger.info(f"Val: {data['val_mask'].sum()}")
        logger.info(f"Test: {data['test_mask'].sum()}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()