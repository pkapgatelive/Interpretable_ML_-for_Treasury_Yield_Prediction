#!/usr/bin/env python3
"""
Data Acquisition Orchestrator

This script coordinates the download of all data sources for the yield curve
forecasting project, including validation and documentation generation.

Usage:
    python scripts/download_all_data.py --start-date 2000-01-01
    python scripts/download_all_data.py --quick  # Download recent data only
    python scripts/download_all_data.py --full   # Complete historical download
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import subprocess
import logging

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.helpers import setup_logging, create_directory

# Setup
app = typer.Typer(help="Data Acquisition Orchestrator for Yield Curve Project")
console = Console()

# Data acquisition scripts
DATA_SCRIPTS = {
    "treasury": "src/data/get_treasury_yield.py",
    "macro": "src/data/get_macro_covariates.py", 
    "ecb": "src/data/get_ecb_yields.py",
    "dictionary": "src/data/generate_data_dictionary.py"
}

# Default configurations
DEFAULT_CONFIG = {
    "quick": {
        "start_date": "2020-01-01",
        "description": "Recent data for quick setup"
    },
    "full": {
        "start_date": "1990-01-01", 
        "description": "Complete historical dataset"
    },
    "custom": {
        "start_date": "2010-01-01",
        "description": "Custom date range"
    }
}


class DataOrchestrator:
    """
    Orchestrates the complete data acquisition pipeline.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize the data orchestrator."""
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def run_script(
        self,
        script_name: str,
        script_path: str,
        args: list,
        description: str
    ) -> Dict[str, Any]:
        """Run a data acquisition script with error handling."""
        console.print(f"\n[blue]🔄 {description}...[/blue]")
        
        start_time = datetime.now()
        
        try:
            # Construct command
            cmd = [sys.executable, script_path] + args
            
            # Run script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root
            )
            
            duration = datetime.now() - start_time
            
            if result.returncode == 0:
                console.print(f"[green]✓ {description} completed successfully[/green]")
                self.logger.info(f"{script_name} completed successfully in {duration}")
                
                return {
                    "status": "success",
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                console.print(f"[red]✗ {description} failed[/red]")
                console.print(f"[red]Error: {result.stderr}[/red]")
                self.logger.error(f"{script_name} failed: {result.stderr}")
                
                return {
                    "status": "failed", 
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
                
        except Exception as e:
            duration = datetime.now() - start_time
            console.print(f"[red]✗ {description} failed with exception: {e}[/red]")
            self.logger.error(f"{script_name} failed with exception: {e}")
            
            return {
                "status": "error",
                "duration": duration,
                "error": str(e)
            }
    
    def download_treasury_data(
        self,
        start_date: str,
        end_date: str,
        format: str = "csv",
        validate: bool = True
    ) -> Dict[str, Any]:
        """Download US Treasury yield curve data."""
        args = [
            "download",
            "--start-date", start_date,
            "--end-date", end_date,
            "--output-dir", str(self.output_dir),
            "--format", format
        ]
        
        if validate:
            args.extend(["--validate", "--save-validation"])
        else:
            args.append("--no-validate")
        
        return self.run_script(
            "treasury",
            DATA_SCRIPTS["treasury"],
            args,
            "Downloading US Treasury yield curves"
        )
    
    def download_macro_data(
        self,
        start_date: str,
        end_date: str,
        format: str = "csv",
        variables: Optional[str] = None
    ) -> Dict[str, Any]:
        """Download macroeconomic covariates."""
        args = [
            "download",
            "--start-date", start_date,
            "--end-date", end_date,
            "--output-dir", str(self.output_dir),
            "--format", format
        ]
        
        if variables:
            args.extend(["--variables", variables])
        
        return self.run_script(
            "macro",
            DATA_SCRIPTS["macro"],
            args,
            "Downloading macroeconomic indicators"
        )
    
    def download_ecb_data(
        self,
        start_date: str,
        end_date: str,
        format: str = "csv",
        countries: str = "DE,FR,IT"
    ) -> Dict[str, Any]:
        """Download ECB yield curve data."""
        args = [
            "download",
            "--start-date", start_date,
            "--end-date", end_date,
            "--output-dir", str(self.output_dir),
            "--format", format,
            "--countries", countries
        ]
        
        return self.run_script(
            "ecb",
            DATA_SCRIPTS["ecb"],
            args,
            "Downloading European yield curves"
        )
    
    def generate_documentation(
        self,
        output_dir: str = "docs",
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """Generate data dictionary documentation."""
        args = [
            "generate",
            "--output-dir", output_dir,
            "--format", format,
            "--include-stats"
        ]
        
        return self.run_script(
            "dictionary",
            DATA_SCRIPTS["dictionary"],
            args,
            "Generating data dictionary"
        )
    
    def display_summary(self) -> None:
        """Display summary of download results."""
        console.print("\n" + "="*60)
        console.print("[bold blue]📊 Data Acquisition Summary[/bold blue]")
        console.print("="*60)
        
        # Create summary table
        table = Table(title="Download Results")
        table.add_column("Data Source", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Duration", style="green")
        table.add_column("Notes", style="yellow")
        
        total_duration = timedelta()
        success_count = 0
        
        for script_name, result in self.results.items():
            if result["status"] == "success":
                status_icon = "✓ Success"
                status_style = "green"
                success_count += 1
            elif result["status"] == "failed":
                status_icon = "✗ Failed"
                status_style = "red"
            else:
                status_icon = "⚠ Error"
                status_style = "yellow"
            
            duration_str = str(result["duration"]).split(".")[0]  # Remove microseconds
            total_duration += result["duration"]
            
            notes = ""
            if result["status"] != "success" and "stderr" in result:
                notes = result["stderr"][:50] + "..." if len(result["stderr"]) > 50 else result["stderr"]
            
            table.add_row(
                script_name.title(),
                f"[{status_style}]{status_icon}[/{status_style}]",
                duration_str,
                notes
            )
        
        console.print(table)
        
        # Overall statistics
        total_duration_str = str(total_duration).split(".")[0]
        console.print(f"\n[bold]Overall Results:[/bold]")
        console.print(f"• Successful downloads: {success_count}/{len(self.results)}")
        console.print(f"• Total duration: {total_duration_str}")
        console.print(f"• Output directory: {self.output_dir}")
        
        # Next steps
        if success_count == len(self.results):
            console.print("\n[green]🎉 All data sources downloaded successfully![/green]")
            console.print("\n[bold yellow]Next Steps:[/bold yellow]")
            console.print("1. Explore the downloaded data in notebooks/")
            console.print("2. Review the data dictionary in docs/")
            console.print("3. Run data processing and feature engineering")
            console.print("4. Begin model development")
        else:
            console.print("\n[red]⚠ Some downloads failed. Check the error messages above.[/red]")
            console.print("\n[bold yellow]Troubleshooting:[/bold yellow]")
            console.print("1. Check your FRED API key: export FRED_API_KEY='your_key'")
            console.print("2. Verify internet connectivity")
            console.print("3. Try downloading individual sources separately")


@app.command()
def quick(
    format: str = typer.Option(
        "csv",
        "--format",
        "-f", 
        help="Output format (csv or parquet)"
    ),
    output_dir: str = typer.Option(
        "data/raw",
        "--output-dir",
        "-o",
        help="Output directory"
    ),
    skip_ecb: bool = typer.Option(
        False,
        "--skip-ecb",
        help="Skip ECB data download (faster)"
    )
):
    """Quick download of recent data (from 2020)."""
    config = DEFAULT_CONFIG["quick"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    console.print(Panel.fit(
        f"[bold blue]🚀 Quick Data Download[/bold blue]\n"
        f"Date range: {config['start_date']} to {end_date}\n"
        f"Description: {config['description']}",
        border_style="blue"
    ))
    
    orchestrator = DataOrchestrator(output_dir)
    setup_logging(log_level="INFO")
    
    # Download data sources
    orchestrator.results["treasury"] = orchestrator.download_treasury_data(
        config["start_date"], end_date, format
    )
    
    orchestrator.results["macro"] = orchestrator.download_macro_data(
        config["start_date"], end_date, format
    )
    
    if not skip_ecb:
        orchestrator.results["ecb"] = orchestrator.download_ecb_data(
            config["start_date"], end_date, format
        )
    
    orchestrator.results["dictionary"] = orchestrator.generate_documentation()
    
    orchestrator.display_summary()


@app.command()
def full(
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Output format (csv or parquet)"
    ),
    output_dir: str = typer.Option(
        "data/raw",
        "--output-dir", 
        "-o",
        help="Output directory"
    )
):
    """Full download of historical data (from 1990)."""
    config = DEFAULT_CONFIG["full"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    console.print(Panel.fit(
        f"[bold blue]📚 Full Historical Download[/bold blue]\n"
        f"Date range: {config['start_date']} to {end_date}\n"
        f"Description: {config['description']}\n"
        f"[yellow]Note: This may take several minutes[/yellow]",
        border_style="blue"
    ))
    
    orchestrator = DataOrchestrator(output_dir)
    setup_logging(log_level="INFO")
    
    # Download all data sources
    orchestrator.results["treasury"] = orchestrator.download_treasury_data(
        config["start_date"], end_date, format
    )
    
    orchestrator.results["macro"] = orchestrator.download_macro_data(
        config["start_date"], end_date, format
    )
    
    orchestrator.results["ecb"] = orchestrator.download_ecb_data(
        config["start_date"], end_date, format
    )
    
    orchestrator.results["dictionary"] = orchestrator.generate_documentation()
    
    orchestrator.display_summary()


@app.command()  
def custom(
    start_date: str = typer.Option(
        "2010-01-01",
        "--start-date",
        "-s",
        help="Start date (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(
        datetime.now().strftime("%Y-%m-%d"),
        "--end-date",
        "-e", 
        help="End date (YYYY-MM-DD)"
    ),
    format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Output format (csv or parquet)"
    ),
    output_dir: str = typer.Option(
        "data/raw",
        "--output-dir",
        "-o",
        help="Output directory"
    ),
    sources: str = typer.Option(
        "treasury,macro,ecb",
        "--sources",
        help="Comma-separated list of data sources"
    )
):
    """Custom download with specified parameters."""
    console.print(Panel.fit(
        f"[bold blue]⚙️ Custom Data Download[/bold blue]\n"
        f"Date range: {start_date} to {end_date}\n"
        f"Sources: {sources}\n"
        f"Format: {format.upper()}",
        border_style="blue"
    ))
    
    orchestrator = DataOrchestrator(output_dir)
    setup_logging(log_level="INFO")
    
    source_list = [s.strip() for s in sources.split(",")]
    
    # Download requested sources
    if "treasury" in source_list:
        orchestrator.results["treasury"] = orchestrator.download_treasury_data(
            start_date, end_date, format
        )
    
    if "macro" in source_list:
        orchestrator.results["macro"] = orchestrator.download_macro_data(
            start_date, end_date, format
        )
    
    if "ecb" in source_list:
        orchestrator.results["ecb"] = orchestrator.download_ecb_data(
            start_date, end_date, format
        )
    
    # Always generate documentation
    orchestrator.results["dictionary"] = orchestrator.generate_documentation()
    
    orchestrator.display_summary()


@app.command()
def info():
    """Display information about data acquisition options."""
    console.print("[bold blue]Data Acquisition Orchestrator[/bold blue]")
    
    console.print("\n[bold yellow]Available Commands:[/bold yellow]")
    console.print("• [cyan]quick[/cyan]  - Download recent data (2020-present)")
    console.print("• [cyan]full[/cyan]   - Download complete historical data (1990-present)")
    console.print("• [cyan]custom[/cyan] - Download with custom parameters")
    
    console.print("\n[bold yellow]Data Sources:[/bold yellow]")
    console.print("• [cyan]treasury[/cyan] - US Treasury yield curves (FRED)")
    console.print("• [cyan]macro[/cyan]    - Macroeconomic indicators (FRED)")
    console.print("• [cyan]ecb[/cyan]      - European government bond yields (ECB)")
    
    console.print("\n[bold yellow]Requirements:[/bold yellow]")
    console.print("• FRED API key (set FRED_API_KEY environment variable)")
    console.print("• Internet connectivity")
    console.print("• Required Python packages (see requirements.txt)")
    
    console.print("\n[bold yellow]Usage Examples:[/bold yellow]")
    console.print("  [dim]python scripts/download_all_data.py quick[/dim]")
    console.print("  [dim]python scripts/download_all_data.py full --format parquet[/dim]")
    console.print("  [dim]python scripts/download_all_data.py custom --start-date 2015-01-01[/dim]")


if __name__ == "__main__":
    app() 