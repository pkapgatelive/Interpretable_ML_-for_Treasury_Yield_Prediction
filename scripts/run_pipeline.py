#!/usr/bin/env python3
"""
Master pipeline script for yield curve forecasting project.

This script orchestrates the complete machine learning pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training and evaluation
- Results analysis and reporting

Usage:
    python scripts/run_pipeline.py --full
    python scripts/run_pipeline.py --steps data,features,models
    python scripts/run_pipeline.py --config config/config.yaml --models random_forest,xgboost
"""

import argparse
import logging
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.helpers import setup_logging, create_directory
from config import load_config

# Setup logging
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete yield curve forecasting pipeline"
    )
    
    # Pipeline control
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete pipeline (all steps)"
    )
    
    parser.add_argument(
        "--steps",
        type=str,
        default="data,features,models,evaluation",
        help="Comma-separated list of pipeline steps to run"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="random_forest,xgboost,lstm",
        help="Comma-separated list of models to train"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_config.yaml", 
        help="Path to model configuration file"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline execution if a step fails"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run model training in parallel when possible"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for pipeline outputs"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file"
    )
    
    return parser.parse_args()


class PipelineRunner:
    """
    Main pipeline orchestration class.
    
    This class manages the execution of different pipeline steps
    and handles dependencies, error handling, and logging.
    """
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.start_time = datetime.now()
        self.results = {}
        
        # Create output directories
        self.output_dir = Path(args.output_dir)
        create_directory(self.output_dir)
        create_directory(self.output_dir / "logs")
        create_directory(self.output_dir / "data")
        create_directory(self.output_dir / "models")
        create_directory(self.output_dir / "reports")
    
    def run_command(self, command: List[str], step_name: str) -> bool:
        """
        Execute a command and log the results.
        
        Parameters
        ----------
        command : List[str]
            Command to execute
        step_name : str
            Name of the pipeline step
            
        Returns
        -------
        bool
            True if command succeeded, False otherwise
        """
        logger.info(f"Running {step_name}: {' '.join(command)}")
        
        if self.args.dry_run:
            logger.info(f"DRY RUN: Would execute {' '.join(command)}")
            return True
        
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                cwd=project_root
            )
            
            execution_time = time.time() - start_time
            
            # Log success
            logger.info(f"{step_name} completed in {execution_time:.2f}s")
            
            # Save logs
            log_file = self.output_dir / "logs" / f"{step_name}.log"
            with open(log_file, "w") as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Execution time: {execution_time:.2f}s\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            self.results[step_name] = {
                "status": "success",
                "execution_time": execution_time,
                "command": command
            }
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"{step_name} failed with return code {e.returncode}")
            logger.error(f"Error: {e.stderr}")
            
            # Save error logs
            log_file = self.output_dir / "logs" / f"{step_name}_error.log"
            with open(log_file, "w") as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Return code: {e.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(e.stdout)
                f.write("\nSTDERR:\n")
                f.write(e.stderr)
            
            self.results[step_name] = {
                "status": "failed",
                "error": str(e),
                "command": command
            }
            
            return False
    
    def run_data_step(self) -> bool:
        """Run data loading and preprocessing step."""
        logger.info("=== STEP 1: Data Loading and Preprocessing ===")
        
        # TODO: Create data loading script
        command = [
            "python", "scripts/data_preprocessing.py",
            "--config", self.args.config,
            "--output", str(self.output_dir / "data")
        ]
        
        return self.run_command(command, "data_preprocessing")
    
    def run_features_step(self) -> bool:
        """Run feature engineering step."""
        logger.info("=== STEP 2: Feature Engineering ===")
        
        # TODO: Create feature engineering script
        command = [
            "python", "scripts/feature_engineering.py",
            "--config", self.args.config,
            "--input", str(self.output_dir / "data"),
            "--output", str(self.output_dir / "data")
        ]
        
        return self.run_command(command, "feature_engineering")
    
    def run_models_step(self) -> bool:
        """Run model training step."""
        logger.info("=== STEP 3: Model Training ===")
        
        models = self.args.models.split(",")
        success = True
        
        if self.args.parallel and len(models) > 1:
            # TODO: Implement parallel model training
            logger.info("Running models in parallel...")
            # For now, run sequentially
            
        for model in models:
            model = model.strip()
            logger.info(f"Training {model} model...")
            
            command = [
                "python", "scripts/train_model.py",
                "--model", model,
                "--config", self.args.config,
                "--model-config", self.args.model_config,
                "--output-dir", str(self.output_dir / "models")
            ]
            
            step_success = self.run_command(command, f"train_{model}")
            if not step_success and not self.args.continue_on_error:
                return False
            success = success and step_success
        
        return success
    
    def run_evaluation_step(self) -> bool:
        """Run model evaluation step."""
        logger.info("=== STEP 4: Model Evaluation ===")
        
        command = [
            "python", "scripts/evaluate_model.py",
            "--config", self.args.config,
            "--models-dir", str(self.output_dir / "models"),
            "--output", str(self.output_dir / "reports")
        ]
        
        return self.run_command(command, "model_evaluation")
    
    def run_reporting_step(self) -> bool:
        """Generate final reports and visualizations."""
        logger.info("=== STEP 5: Report Generation ===")
        
        # TODO: Create reporting script
        command = [
            "python", "scripts/generate_report.py",
            "--config", self.args.config,
            "--results-dir", str(self.output_dir),
            "--output", str(self.output_dir / "reports")
        ]
        
        return self.run_command(command, "report_generation")
    
    def run_pipeline(self) -> bool:
        """
        Run the complete pipeline based on specified steps.
        
        Returns
        -------
        bool
            True if pipeline completed successfully
        """
        logger.info(f"Starting pipeline at {self.start_time}")
        logger.info(f"Configuration: {self.args.config}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Determine steps to run
        if self.args.full:
            steps = ["data", "features", "models", "evaluation", "reporting"]
        else:
            steps = [s.strip() for s in self.args.steps.split(",")]
        
        logger.info(f"Pipeline steps: {steps}")
        
        # Execute steps
        success = True
        for step in steps:
            try:
                if step == "data":
                    step_success = self.run_data_step()
                elif step == "features":
                    step_success = self.run_features_step()
                elif step == "models":
                    step_success = self.run_models_step()
                elif step == "evaluation":
                    step_success = self.run_evaluation_step()
                elif step == "reporting":
                    step_success = self.run_reporting_step()
                else:
                    logger.warning(f"Unknown step: {step}")
                    continue
                
                if not step_success:
                    logger.error(f"Step {step} failed")
                    if not self.args.continue_on_error:
                        success = False
                        break
                    success = False
                    
            except Exception as e:
                logger.error(f"Unexpected error in step {step}: {e}")
                if not self.args.continue_on_error:
                    success = False
                    break
                success = False
        
        # Generate pipeline summary
        self.generate_summary()
        
        total_time = datetime.now() - self.start_time
        if success:
            logger.info(f"Pipeline completed successfully in {total_time}")
        else:
            logger.error(f"Pipeline completed with errors in {total_time}")
        
        return success
    
    def generate_summary(self):
        """Generate pipeline execution summary."""
        summary = {
            "pipeline_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": str(datetime.now() - self.start_time),
                "config_file": self.args.config,
                "output_directory": str(self.output_dir)
            },
            "execution_results": self.results,
            "models_trained": self.args.models.split(","),
            "steps_completed": list(self.results.keys())
        }
        
        # Save summary
        summary_file = self.output_dir / "pipeline_summary.yaml"
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Pipeline summary saved to {summary_file}")


def main():
    """Main pipeline execution function."""
    args = parse_arguments()
    
    # Setup logging
    log_file = args.log_file or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(log_level="INFO", log_file=log_file)
    
    logger.info("="*60)
    logger.info("YIELD CURVE FORECASTING PIPELINE")
    logger.info("="*60)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return False
    
    # Create and run pipeline
    pipeline = PipelineRunner(config, args)
    success = pipeline.run_pipeline()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 