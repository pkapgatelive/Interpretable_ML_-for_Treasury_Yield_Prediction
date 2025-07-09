#!/usr/bin/env python3
"""
US Treasury Yield Curve Data Acquisition Script

This script downloads daily Treasury par-yield curves from the Federal Reserve
Economic Data (FRED) API, validates the data, and stores it with proper formatting.

Usage:
    python get_treasury_yield.py --start-date 1990-01-01 --end-date 2025-01-01
    python get_treasury_yield.py --tenors 1M,3M,6M,1Y,2Y,5Y,10Y,30Y --format parquet
    python get_treasury_yield.py --output-dir data/raw --validate
"""

import os
import sys
from datetime import datetime, date
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
from src.utils.constants import FRED_SERIES_IDS

# Setup
app = typer.Typer(help="US Treasury Yield Curve Data Acquisition")
console = Console()
logger = logging.getLogger(__name__)

# FRED Series Mapping for Treasury Yields
TREASURY_SERIES = {
    "1M": "DGS1MO",    # 1-Month Treasury
    "3M": "DGS3MO",    # 3-Month Treasury  
    "6M": "DGS6MO",    # 6-Month Treasury
    "1Y": "DGS1",      # 1-Year Treasury
    "2Y": "DGS2",      # 2-Year Treasury
    "3Y": "DGS3",      # 3-Year Treasury
    "5Y": "DGS5",      # 5-Year Treasury
    "7Y": "DGS7",      # 7-Year Treasury
    "10Y": "DGS10",    # 10-Year Treasury
    "20Y": "DGS20",    # 20-Year Treasury
    "30Y": "DGS30",    # 30-Year Treasury
}

# Data Schema Definition
treasury_schema = DataFrameSchema({
    "date": Column(pa.DateTime, nullable=False, unique=True),
    **{
        tenor: Column(
            pa.Float, 
            nullable=True,  # FRED data can have missing values
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(50.0)  # Reasonable upper bound for yields
            ],
            description=f"{tenor} Treasury Constant Maturity Rate (%)"
        )
        for tenor in TREASURY_SERIES.keys()
    }
})


class TreasuryDataLoader:
    """
    Class for loading and processing US Treasury yield curve data from FRED.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Treasury data loader.
        
        Parameters
        ----------
        api_key : Optional[str]
            FRED API key. If None, will try to load from environment.
        """
        if api_key is None:
            api_key = os.getenv("FRED_API_KEY")
            
        if not api_key:
            console.print(
                "[red]Error: FRED API key not found. Please set FRED_API_KEY environment variable or pass --api-key[/red]"
            )
            raise typer.Exit(1)
            
        try:
            self.fred = Fred(api_key=api_key)
            logger.info("Successfully initialized FRED API client")
        except Exception as e:
            console.print(f"[red]Error initializing FRED API: {e}[/red]")
            raise typer.Exit(1)
    
    def download_yield_data(
        self,
        tenors: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Download Treasury yield data for specified tenors and date range.
        
        Parameters
        ----------
        tenors : List[str]
            List of tenors to download (e.g., ['1M', '3M', '1Y', '10Y'])
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns
        -------
        pd.DataFrame
            DataFrame with dates as index and tenors as columns
        """
        console.print(f"[blue]Downloading Treasury yield data for tenors: {', '.join(tenors)}[/blue]")
        
        data_dict = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for tenor in tenors:
                if tenor not in TREASURY_SERIES:
                    console.print(f"[yellow]Warning: Unknown tenor {tenor}, skipping[/yellow]")
                    continue
                    
                task = progress.add_task(f"Downloading {tenor} yields...", total=None)
                
                try:
                    series_id = TREASURY_SERIES[tenor]
                    series_data = self.fred.get_series(
                        series_id,
                        start=start_date,
                        end=end_date
                    )
                    
                    if not series_data.empty:
                        data_dict[tenor] = series_data
                        logger.info(f"Downloaded {len(series_data)} observations for {tenor}")
                    else:
                        console.print(f"[yellow]Warning: No data found for {tenor}[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]Error downloading {tenor}: {e}[/red]")
                    continue
                    
                progress.update(task, completed=True)
        
        if not data_dict:
            console.print("[red]No data was successfully downloaded[/red]")
            raise typer.Exit(1)
        
        # Combine all series into a single DataFrame
        df = pd.DataFrame(data_dict)
        df.index.name = 'date'
        df.reset_index(inplace=True)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        console.print(f"[green]Successfully downloaded {len(df)} daily observations[/green]")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the downloaded Treasury yield data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Treasury yield data to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation results and statistics
        """
        console.print("[blue]Validating Treasury yield data...[/blue]")
        
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Basic pandera validation
            validated_df = treasury_schema.validate(df, lazy=True)
            console.print("[green]✓ Schema validation passed[/green]")
            
        except pa.errors.SchemaErrors as e:
            validation_results["is_valid"] = False
            validation_results["errors"].extend([str(error) for error in e.schema_errors])
            console.print(f"[red]✗ Schema validation failed: {len(e.schema_errors)} errors[/red]")
        
        # Additional validation checks
        try:
            # Check for monotonic dates
            if not df['date'].is_monotonic_increasing:
                validation_results["warnings"].append("Dates are not monotonic")
                
            # Check data coverage
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()
            
            if total_missing > 0:
                validation_results["warnings"].append(f"Found {total_missing} missing values")
                validation_results["statistics"]["missing_by_tenor"] = missing_data.to_dict()
            
            # Check yield curve shape (basic sanity check)
            tenor_order = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
            available_tenors = [t for t in tenor_order if t in df.columns]
            
            if len(available_tenors) >= 3:
                # Check if yield curve is generally upward sloping on average
                avg_yields = df[available_tenors].mean()
                inversions = 0
                for i in range(len(available_tenors) - 1):
                    if avg_yields[available_tenors[i]] > avg_yields[available_tenors[i+1]]:
                        inversions += 1
                        
                inversion_rate = inversions / (len(available_tenors) - 1)
                validation_results["statistics"]["average_inversion_rate"] = inversion_rate
                
                if inversion_rate > 0.5:
                    validation_results["warnings"].append(
                        f"High inversion rate: {inversion_rate:.2%} of tenor pairs are inverted on average"
                    )
            
            # Date range statistics
            validation_results["statistics"]["date_range"] = {
                "start": df['date'].min().strftime('%Y-%m-%d'),
                "end": df['date'].max().strftime('%Y-%m-%d'),
                "total_days": len(df),
                "unique_dates": df['date'].nunique()
            }
            
            # Yield statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            validation_results["statistics"]["yield_stats"] = {
                "mean_yields": df[numeric_cols].mean().to_dict(),
                "std_yields": df[numeric_cols].std().to_dict(),
                "min_yields": df[numeric_cols].min().to_dict(),
                "max_yields": df[numeric_cols].max().to_dict()
            }
            
        except Exception as e:
            validation_results["errors"].append(f"Validation error: {str(e)}")
            validation_results["is_valid"] = False
        
        # Print validation summary
        if validation_results["is_valid"]:
            console.print("[green]✓ Data validation completed successfully[/green]")
        else:
            console.print("[red]✗ Data validation failed[/red]")
            
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
    tenors: str = typer.Option(
        "1M,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,20Y,30Y",
        "--tenors",
        "-t",
        help="Comma-separated list of tenors to download"
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
    ),
    save_validation: bool = typer.Option(
        False,
        "--save-validation",
        help="Save validation report to file"
    )
):
    """
    Download US Treasury yield curve data from FRED.
    """
    # Setup logging
    setup_logging(log_level="INFO")
    
    console.print("[bold blue]US Treasury Yield Curve Data Acquisition[/bold blue]")
    console.print(f"Date range: {start_date} to {end_date}")
    
    # Parse tenors
    tenor_list = [t.strip().upper() for t in tenors.split(",")]
    console.print(f"Tenors: {', '.join(tenor_list)}")
    
    # Create output directory
    output_path = Path(output_dir)
    create_directory(output_path)
    
    try:
        # Initialize data loader
        loader = TreasuryDataLoader(api_key=api_key)
        
        # Download data
        df = loader.download_yield_data(tenor_list, start_date, end_date)
        
        # Validate data
        validation_results = None
        if validate:
            validation_results = loader.validate_data(df)
            
            if save_validation:
                validation_path = output_path / f"treasury_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                import json
                with open(validation_path, 'w') as f:
                    json.dump(validation_results, f, indent=2, default=str)
                console.print(f"[blue]Validation report saved to {validation_path}[/blue]")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        if output_format.lower() == "parquet":
            filename = f"yieldcurve_us_{timestamp}.parquet"
            filepath = output_path / filename
            df.to_parquet(filepath, index=False)
        else:
            filename = f"yieldcurve_us_{timestamp}.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=False)
        
        console.print(f"[green]✓ Data saved to {filepath}[/green]")
        
        # Display summary table
        table = Table(title="Download Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Records Downloaded", str(len(df)))
        table.add_row("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        table.add_row("Tenors", str(len([c for c in df.columns if c != 'date'])))
        table.add_row("File Size", f"{filepath.stat().st_size / 1024:.1f} KB")
        table.add_row("Output Format", output_format.upper())
        
        if validation_results:
            table.add_row("Validation", "✓ Passed" if validation_results["is_valid"] else "✗ Failed")
            if validation_results["warnings"]:
                table.add_row("Warnings", str(len(validation_results["warnings"])))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Download failed: {e}")
        raise typer.Exit(1)


@app.command()
def info():
    """
    Display information about available Treasury yield tenors and FRED series.
    """
    console.print("[bold blue]US Treasury Yield Curve Data Information[/bold blue]")
    
    table = Table(title="Available Treasury Yield Tenors")
    table.add_column("Tenor", style="cyan")
    table.add_column("FRED Series ID", style="green")
    table.add_column("Description", style="white")
    
    descriptions = {
        "1M": "1-Month Treasury Constant Maturity Rate",
        "3M": "3-Month Treasury Constant Maturity Rate", 
        "6M": "6-Month Treasury Constant Maturity Rate",
        "1Y": "1-Year Treasury Constant Maturity Rate",
        "2Y": "2-Year Treasury Constant Maturity Rate",
        "3Y": "3-Year Treasury Constant Maturity Rate",
        "5Y": "5-Year Treasury Constant Maturity Rate",
        "7Y": "7-Year Treasury Constant Maturity Rate",
        "10Y": "10-Year Treasury Constant Maturity Rate",
        "20Y": "20-Year Treasury Constant Maturity Rate",
        "30Y": "30-Year Treasury Constant Maturity Rate"
    }
    
    for tenor, series_id in TREASURY_SERIES.items():
        table.add_row(tenor, series_id, descriptions.get(tenor, "Treasury Constant Maturity Rate"))
    
    console.print(table)
    
    console.print("\n[bold yellow]Usage Examples:[/bold yellow]")
    console.print("• Download all tenors (default):")
    console.print("  [dim]python get_treasury_yield.py download[/dim]")
    console.print("• Download specific tenors:")
    console.print("  [dim]python get_treasury_yield.py download --tenors 2Y,5Y,10Y,30Y[/dim]")
    console.print("• Download historical data:")
    console.print("  [dim]python get_treasury_yield.py download --start-date 2000-01-01 --end-date 2023-12-31[/dim]")
    console.print("• Save as Parquet:")
    console.print("  [dim]python get_treasury_yield.py download --format parquet[/dim]")


if __name__ == "__main__":
    app() 