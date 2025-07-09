#!/usr/bin/env python3
"""
Macroeconomic Covariates Data Acquisition Script

This script downloads macroeconomic indicators from FRED that serve as covariates
for yield curve forecasting and monetary policy analysis.

Usage:
    python get_macro_covariates.py download --start-date 1990-01-01
    python get_macro_covariates.py download --variables FEDFUNDS,CPIAUCSL,NAPM --format parquet
    python get_macro_covariates.py info
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

import typer
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from fredapi import Fred
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.helpers import setup_logging, create_directory

# Setup
app = typer.Typer(help="Macroeconomic Covariates Data Acquisition")
console = Console()
logger = logging.getLogger(__name__)

# FRED Series Mapping for Macro Variables
MACRO_SERIES = {
    # Monetary Policy
    "FEDFUNDS": {"name": "Federal Funds Rate", "freq": "monthly", "unit": "percent"},
    "DFF": {"name": "Daily Federal Funds Rate", "freq": "daily", "unit": "percent"},
    "TB3MS": {"name": "3-Month Treasury Bill Rate", "freq": "monthly", "unit": "percent"},
    
    # Inflation Indicators
    "CPIAUCSL": {"name": "Consumer Price Index", "freq": "monthly", "unit": "index"},
    "CPILFESL": {"name": "Core CPI", "freq": "monthly", "unit": "index"},
    "PCEPILFE": {"name": "Core PCE Price Index", "freq": "monthly", "unit": "index"},
    "T5YIE": {"name": "5-Year Breakeven Inflation Rate", "freq": "daily", "unit": "percent"},
    "T10YIE": {"name": "10-Year Breakeven Inflation Rate", "freq": "daily", "unit": "percent"},
    
    # Economic Activity
    "INDPRO": {"name": "Industrial Production Index", "freq": "monthly", "unit": "index"},
    "PAYEMS": {"name": "Nonfarm Payrolls", "freq": "monthly", "unit": "thousands"},
    "UNRATE": {"name": "Unemployment Rate", "freq": "monthly", "unit": "percent"},
    "NAPM": {"name": "ISM Manufacturing PMI", "freq": "monthly", "unit": "index"},
    "UMCSENT": {"name": "Consumer Sentiment", "freq": "monthly", "unit": "index"},
    
    # Financial Markets
    "DEXUSEU": {"name": "USD/EUR Exchange Rate", "freq": "daily", "unit": "rate"},
    "DEXJPUS": {"name": "JPY/USD Exchange Rate", "freq": "daily", "unit": "rate"},
    "VIXCLS": {"name": "VIX Volatility Index", "freq": "daily", "unit": "index"},
    "SP500": {"name": "S&P 500", "freq": "daily", "unit": "index"},
    
    # Credit Conditions
    "BAMLC0A0CM": {"name": "Corporate Bond Spread", "freq": "daily", "unit": "percent"},
    "TEDRATE": {"name": "TED Spread", "freq": "daily", "unit": "percent"},
    
    # Housing
    "HOUST": {"name": "Housing Starts", "freq": "monthly", "unit": "thousands"},
    "CSUSHPISA": {"name": "Case-Shiller Home Price Index", "freq": "monthly", "unit": "index"},
}


class MacroDataLoader:
    """
    Class for loading and processing macroeconomic data from FRED.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the macro data loader."""
        if api_key is None:
            api_key = os.getenv("FRED_API_KEY")
            
        if not api_key:
            console.print(
                "[red]Error: FRED API key not found. Please set FRED_API_KEY environment variable[/red]"
            )
            raise typer.Exit(1)
            
        try:
            self.fred = Fred(api_key=api_key)
            logger.info("Successfully initialized FRED API client")
        except Exception as e:
            console.print(f"[red]Error initializing FRED API: {e}[/red]")
            raise typer.Exit(1)
    
    def download_macro_data(
        self,
        variables: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download macroeconomic data for specified variables."""
        console.print(f"[blue]Downloading macro data for: {', '.join(variables)}[/blue]")
        
        data_dict = {}
        metadata_dict = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for variable in variables:
                if variable not in MACRO_SERIES:
                    console.print(f"[yellow]Warning: Unknown variable {variable}, skipping[/yellow]")
                    continue
                    
                task = progress.add_task(f"Downloading {variable}...", total=None)
                
                try:
                    series_data = self.fred.get_series(
                        variable,
                        start=start_date,
                        end=end_date
                    )
                    
                    if not series_data.empty:
                        data_dict[variable] = series_data
                        metadata_dict[variable] = MACRO_SERIES[variable]
                        logger.info(f"Downloaded {len(series_data)} observations for {variable}")
                    else:
                        console.print(f"[yellow]Warning: No data found for {variable}[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]Error downloading {variable}: {e}[/red]")
                    continue
                    
                progress.update(task, completed=True)
        
        if not data_dict:
            console.print("[red]No data was successfully downloaded[/red]")
            raise typer.Exit(1)
        
        # Combine all series into a single DataFrame
        df = pd.DataFrame(data_dict)
        df.index.name = 'date'
        df.reset_index(inplace=True)
        df = df.sort_values('date').reset_index(drop=True)
        
        console.print(f"[green]Successfully downloaded {len(df)} observations[/green]")
        
        # Store metadata for later use
        self.metadata = metadata_dict
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the downloaded macro data."""
        console.print("[blue]Validating macroeconomic data...[/blue]")
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Basic validation checks
            if not df['date'].is_monotonic_increasing:
                validation_results["warnings"].append("Dates are not monotonic")
            
            # Check for missing data
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()
            
            if total_missing > 0:
                validation_results["warnings"].append(f"Found {total_missing} missing values")
                validation_results["statistics"]["missing_by_variable"] = missing_data.to_dict()
            
            # Date range statistics
            validation_results["statistics"]["date_range"] = {
                "start": df['date'].min().strftime('%Y-%m-%d'),
                "end": df['date'].max().strftime('%Y-%m-%d'),
                "total_observations": len(df),
                "unique_dates": df['date'].nunique()
            }
            
            # Variable statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            validation_results["statistics"]["variable_stats"] = {
                "mean": df[numeric_cols].mean().to_dict(),
                "std": df[numeric_cols].std().to_dict(),
                "min": df[numeric_cols].min().to_dict(),
                "max": df[numeric_cols].max().to_dict()
            }
            
            console.print("[green]✓ Data validation completed successfully[/green]")
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False
            console.print(f"[red]✗ Data validation failed: {e}[/red]")
        
        if validation_results["warnings"]:
            console.print(f"[yellow]⚠ {len(validation_results['warnings'])} warnings found[/yellow]")
            
        return validation_results


@app.command()
def download(
    start_date: str = typer.Option(
        "1990-01-01",
        "--start-date",
        "-s",
        help="Start date for data download (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(
        datetime.now().strftime("%Y-%m-%d"),
        "--end-date", 
        "-e",
        help="End date for data download (YYYY-MM-DD)"
    ),
    variables: str = typer.Option(
        "FEDFUNDS,CPIAUCSL,NAPM,UNRATE,VIXCLS",
        "--variables",
        "-v",
        help="Comma-separated list of FRED series IDs"
    ),
    output_dir: str = typer.Option(
        "data/raw",
        "--output-dir",
        "-o",
        help="Output directory for data files"
    ),
    output_format: str = typer.Option(
        "csv",
        "--format",
        "-f",
        help="Output format (csv or parquet)"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        help="FRED API key (if not set in environment)"
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate downloaded data"
    )
):
    """Download macroeconomic covariates from FRED."""
    setup_logging(log_level="INFO")
    
    console.print("[bold blue]Macroeconomic Covariates Data Acquisition[/bold blue]")
    console.print(f"Date range: {start_date} to {end_date}")
    
    # Parse variables
    var_list = [v.strip().upper() for v in variables.split(",")]
    console.print(f"Variables: {', '.join(var_list)}")
    
    # Create output directory
    output_path = Path(output_dir)
    create_directory(output_path)
    
    try:
        # Initialize data loader
        loader = MacroDataLoader(api_key=api_key)
        
        # Download data
        df = loader.download_macro_data(var_list, start_date, end_date)
        
        # Validate data
        if validate:
            validation_results = loader.validate_data(df)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        if output_format.lower() == "parquet":
            filename = f"macro_fred_{timestamp}.parquet"
            filepath = output_path / filename
            df.to_parquet(filepath, index=False)
        else:
            filename = f"macro_fred_{timestamp}.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=False)
        
        console.print(f"[green]✓ Data saved to {filepath}[/green]")
        
        # Display summary
        table = Table(title="Download Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Records Downloaded", str(len(df)))
        table.add_row("Variables", str(len([c for c in df.columns if c != 'date'])))
        table.add_row("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        table.add_row("File Size", f"{filepath.stat().st_size / 1024:.1f} KB")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Display information about available macro variables."""
    console.print("[bold blue]Available Macroeconomic Variables[/bold blue]")
    
    # Group by category
    categories = {
        "Monetary Policy": ["FEDFUNDS", "DFF", "TB3MS"],
        "Inflation": ["CPIAUCSL", "CPILFESL", "PCEPILFE", "T5YIE", "T10YIE"],
        "Economic Activity": ["INDPRO", "PAYEMS", "UNRATE", "NAPM", "UMCSENT"],
        "Financial Markets": ["DEXUSEU", "DEXJPUS", "VIXCLS", "SP500"],
        "Credit": ["BAMLC0A0CM", "TEDRATE"],
        "Housing": ["HOUST", "CSUSHPISA"]
    }
    
    for category, series_list in categories.items():
        table = Table(title=f"{category} Variables")
        table.add_column("Series ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Frequency", style="green")
        
        for series_id in series_list:
            if series_id in MACRO_SERIES:
                info = MACRO_SERIES[series_id]
                table.add_row(series_id, info["name"], info["freq"])
        
        console.print(table)
        console.print()


if __name__ == "__main__":
    app() 