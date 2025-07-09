#!/usr/bin/env python3
"""
Data Dictionary Generator

This script generates comprehensive data dictionary documentation for all
datasets used in the yield curve forecasting project.

Usage:
    python generate_data_dictionary.py --output-dir docs/
    python generate_data_dictionary.py --format markdown --include-stats
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.constants import FRED_SERIES_IDS
from src.utils.helpers import setup_logging, create_directory

# Setup
app = typer.Typer(help="Data Dictionary Generator for Yield Curve Project")
console = Console()

# Data source metadata
DATA_SOURCES = {
    "FRED": {
        "name": "Federal Reserve Economic Data",
        "provider": "Federal Reserve Bank of St. Louis",
        "url": "https://fred.stlouisfed.org/",
        "license": "Public Domain",
        "update_frequency": "Daily/Monthly (varies by series)",
        "access_method": "REST API",
        "documentation": "https://fred.stlouisfed.org/docs/api/fred/"
    },
    "ECB": {
        "name": "European Central Bank Statistical Data Warehouse",
        "provider": "European Central Bank",
        "url": "https://sdw.ecb.europa.eu/",
        "license": "Creative Commons Attribution 4.0",
        "update_frequency": "Daily",
        "access_method": "SDMX API",
        "documentation": "https://sdw.ecb.europa.eu/help.do"
    }
}

# Variable categories and descriptions
VARIABLE_CATEGORIES = {
    "yield_curves": {
        "description": "Government bond yield curves representing the term structure of interest rates",
        "frequency": "Daily",
        "unit": "Percent per annum",
        "calculation": "Constant maturity yields based on actively traded treasury securities"
    },
    "monetary_policy": {
        "description": "Central bank policy rates and related monetary policy indicators",
        "frequency": "Daily/Monthly",
        "unit": "Percent per annum",
        "calculation": "Official policy rates set by central banks"
    },
    "inflation": {
        "description": "Price level indicators and inflation expectations",
        "frequency": "Monthly/Daily",
        "unit": "Index/Percent",
        "calculation": "Consumer price indices and market-based inflation expectations"
    },
    "economic_activity": {
        "description": "Real economy indicators measuring economic growth and business conditions",
        "frequency": "Monthly",
        "unit": "Index/Thousands/Percent",
        "calculation": "Survey-based and administrative data on economic activity"
    },
    "financial_markets": {
        "description": "Financial market indicators including equity prices, exchange rates, and volatility",
        "frequency": "Daily",
        "unit": "Index/Rate/Percent",
        "calculation": "Market prices and calculated volatility measures"
    },
    "credit_conditions": {
        "description": "Credit market conditions and risk spreads",
        "frequency": "Daily",
        "unit": "Basis points/Percent",
        "calculation": "Credit spreads over government bond yields"
    }
}


class DataDictionaryGenerator:
    """
    Generator for comprehensive data dictionary documentation.
    """
    
    def __init__(self):
        """Initialize the data dictionary generator."""
        self.dictionary = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project": "Interpretable Machine-Learning Models for Yield-Curve Forecasting",
                "version": "1.0.0",
                "description": "Comprehensive data dictionary for yield curve forecasting project"
            },
            "data_sources": DATA_SOURCES,
            "variable_categories": VARIABLE_CATEGORIES,
            "variables": {}
        }
    
    def add_treasury_variables(self) -> None:
        """Add US Treasury yield curve variables to dictionary."""
        treasury_tenors = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
        
        for tenor in treasury_tenors:
            series_id = f"DGS{tenor.replace('M', 'MO')}" if "M" in tenor else f"DGS{tenor.replace('Y', '')}"
            
            self.dictionary["variables"][f"US_TREASURY_{tenor}"] = {
                "variable_name": f"US Treasury {tenor} Yield",
                "series_id": series_id,
                "description": FRED_SERIES_IDS.get(series_id, f"{tenor} Treasury Constant Maturity Rate"),
                "category": "yield_curves",
                "source": "FRED",
                "frequency": "Daily",
                "unit": "Percent per annum",
                "tenor": tenor,
                "country": "United States",
                "currency": "USD",
                "instrument_type": "Government Bond",
                "calculation_method": "Constant Maturity Treasury Rate",
                "seasonal_adjustment": "Not Seasonally Adjusted",
                "data_quality": {
                    "expected_missing": "Weekends and holidays",
                    "typical_range": "0% to 15%",
                    "validation_rules": ["Non-negative", "< 50%", "Term structure consistency"]
                }
            }
    
    def add_macro_variables(self) -> None:
        """Add macroeconomic variables to dictionary."""
        macro_mappings = {
            "FEDFUNDS": {
                "name": "Federal Funds Rate",
                "category": "monetary_policy",
                "frequency": "Monthly",
                "unit": "Percent per annum"
            },
            "DFF": {
                "name": "Daily Federal Funds Rate",
                "category": "monetary_policy", 
                "frequency": "Daily",
                "unit": "Percent per annum"
            },
            "CPIAUCSL": {
                "name": "Consumer Price Index",
                "category": "inflation",
                "frequency": "Monthly",
                "unit": "Index (1982-84=100)"
            },
            "UNRATE": {
                "name": "Unemployment Rate",
                "category": "economic_activity",
                "frequency": "Monthly",
                "unit": "Percent"
            },
            "VIXCLS": {
                "name": "VIX Volatility Index",
                "category": "financial_markets",
                "frequency": "Daily",
                "unit": "Percent"
            },
            "SP500": {
                "name": "S&P 500 Index",
                "category": "financial_markets",
                "frequency": "Daily",
                "unit": "Index"
            }
        }
        
        for series_id, info in macro_mappings.items():
            self.dictionary["variables"][series_id] = {
                "variable_name": info["name"],
                "series_id": series_id,
                "description": FRED_SERIES_IDS.get(series_id, info["name"]),
                "category": info["category"],
                "source": "FRED",
                "frequency": info["frequency"],
                "unit": info["unit"],
                "country": "United States",
                "currency": "USD" if "rate" in info["name"].lower() else None,
                "seasonal_adjustment": "Seasonally Adjusted" if series_id in ["UNRATE", "CPIAUCSL"] else "Not Seasonally Adjusted",
                "data_quality": {
                    "expected_missing": "Limited missing values",
                    "validation_rules": self._get_validation_rules(info["category"])
                }
            }
    
    def add_ecb_variables(self) -> None:
        """Add ECB yield curve variables to dictionary."""
        countries = {"DE": "Germany", "FR": "France", "IT": "Italy"}
        tenors = ["1Y", "2Y", "5Y", "10Y", "30Y"]
        
        for country_code, country_name in countries.items():
            for tenor in tenors:
                var_name = f"ECB_{country_code}_{tenor}"
                
                self.dictionary["variables"][var_name] = {
                    "variable_name": f"{country_name} Government Bond {tenor} Yield",
                    "series_id": f"YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_{tenor}",
                    "description": f"{tenor} yield on {country_name} government bonds",
                    "category": "yield_curves",
                    "source": "ECB",
                    "frequency": "Daily",
                    "unit": "Percent per annum",
                    "tenor": tenor,
                    "country": country_name,
                    "currency": "EUR",
                    "instrument_type": "Government Bond",
                    "calculation_method": "Yield to Maturity",
                    "seasonal_adjustment": "Not Seasonally Adjusted",
                    "data_quality": {
                        "expected_missing": "Weekends and holidays",
                        "typical_range": "-1% to 10%",
                        "validation_rules": ["Can be negative", "< 20%", "Term structure consistency"]
                    }
                }
    
    def _get_validation_rules(self, category: str) -> List[str]:
        """Get validation rules for variable category."""
        rules_map = {
            "monetary_policy": ["Non-negative", "< 25%"],
            "inflation": ["Non-negative", "< 1000 (index)", "< 50% (rate)"],
            "economic_activity": ["Non-negative", "< 100% (unemployment)", "< 200 (index)"],
            "financial_markets": ["Non-negative", "Volatility < 200%"],
            "yield_curves": ["Can be negative", "< 50%", "Term structure consistency"]
        }
        return rules_map.get(category, ["Standard validation"])
    
    def generate_markdown(self, include_stats: bool = False) -> str:
        """Generate markdown documentation."""
        md_content = []
        
        # Header
        md_content.append("# Data Dictionary")
        md_content.append(f"**Project:** {self.dictionary['metadata']['project']}")
        md_content.append(f"**Generated:** {self.dictionary['metadata']['generated_at']}")
        md_content.append(f"**Version:** {self.dictionary['metadata']['version']}")
        md_content.append("")
        md_content.append(self.dictionary['metadata']['description'])
        md_content.append("")
        
        # Data Sources
        md_content.append("## Data Sources")
        md_content.append("")
        for source_id, source_info in self.dictionary['data_sources'].items():
            md_content.append(f"### {source_id}")
            md_content.append(f"**Name:** {source_info['name']}")
            md_content.append(f"**Provider:** {source_info['provider']}")
            md_content.append(f"**URL:** {source_info['url']}")
            md_content.append(f"**License:** {source_info['license']}")
            md_content.append(f"**Update Frequency:** {source_info['update_frequency']}")
            md_content.append(f"**Access Method:** {source_info['access_method']}")
            md_content.append("")
        
        # Variable Categories
        md_content.append("## Variable Categories")
        md_content.append("")
        for cat_id, cat_info in self.dictionary['variable_categories'].items():
            md_content.append(f"### {cat_id.replace('_', ' ').title()}")
            md_content.append(f"**Description:** {cat_info['description']}")
            md_content.append(f"**Frequency:** {cat_info['frequency']}")
            md_content.append(f"**Unit:** {cat_info['unit']}")
            md_content.append("")
        
        # Variables Table
        md_content.append("## Variables")
        md_content.append("")
        md_content.append("| Variable | Description | Source | Frequency | Unit | Country |")
        md_content.append("|----------|-------------|---------|-----------|------|---------|")
        
        for var_id, var_info in self.dictionary['variables'].items():
            country = var_info.get('country', 'N/A')
            md_content.append(
                f"| {var_id} | {var_info['description']} | "
                f"{var_info['source']} | {var_info['frequency']} | "
                f"{var_info['unit']} | {country} |"
            )
        
        md_content.append("")
        
        # Detailed Variable Information
        md_content.append("## Detailed Variable Information")
        md_content.append("")
        
        # Group by category
        by_category = {}
        for var_id, var_info in self.dictionary['variables'].items():
            category = var_info['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((var_id, var_info))
        
        for category, variables in by_category.items():
            md_content.append(f"### {category.replace('_', ' ').title()}")
            md_content.append("")
            
            for var_id, var_info in variables:
                md_content.append(f"#### {var_id}")
                md_content.append(f"- **Description:** {var_info['description']}")
                md_content.append(f"- **Series ID:** {var_info['series_id']}")
                md_content.append(f"- **Source:** {var_info['source']}")
                md_content.append(f"- **Frequency:** {var_info['frequency']}")
                md_content.append(f"- **Unit:** {var_info['unit']}")
                
                if 'country' in var_info:
                    md_content.append(f"- **Country:** {var_info['country']}")
                if 'tenor' in var_info:
                    md_content.append(f"- **Tenor:** {var_info['tenor']}")
                if 'currency' in var_info and var_info['currency']:
                    md_content.append(f"- **Currency:** {var_info['currency']}")
                
                if 'data_quality' in var_info:
                    quality = var_info['data_quality']
                    if 'typical_range' in quality:
                        md_content.append(f"- **Typical Range:** {quality['typical_range']}")
                    if 'validation_rules' in quality:
                        rules = ', '.join(quality['validation_rules'])
                        md_content.append(f"- **Validation Rules:** {rules}")
                
                md_content.append("")
        
        return '\n'.join(md_content)
    
    def save_dictionary(
        self,
        output_dir: str,
        format: str = "markdown",
        include_stats: bool = False
    ) -> Path:
        """Save the data dictionary to file."""
        output_path = Path(output_dir)
        create_directory(output_path)
        
        timestamp = datetime.now().strftime("%Y%m%d")
        
        if format.lower() == "markdown":
            filename = f"data_dictionary_{timestamp}.md"
            filepath = output_path / filename
            
            content = self.generate_markdown(include_stats=include_stats)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
        elif format.lower() == "json":
            filename = f"data_dictionary_{timestamp}.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.dictionary, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filepath


@app.command()
def generate(
    output_dir: str = typer.Option(
        "docs",
        "--output-dir",
        "-o",
        help="Output directory for data dictionary"
    ),
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format (markdown or json)"
    ),
    include_stats: bool = typer.Option(
        False,
        "--include-stats",
        help="Include statistical information in documentation"
    )
):
    """Generate comprehensive data dictionary."""
    setup_logging(log_level="INFO")
    
    console.print("[bold blue]Generating Data Dictionary[/bold blue]")
    
    try:
        # Initialize generator
        generator = DataDictionaryGenerator()
        
        # Add all variable types
        console.print("[blue]Adding Treasury yield variables...[/blue]")
        generator.add_treasury_variables()
        
        console.print("[blue]Adding macroeconomic variables...[/blue]")
        generator.add_macro_variables()
        
        console.print("[blue]Adding ECB yield variables...[/blue]")
        generator.add_ecb_variables()
        
        # Save dictionary
        console.print(f"[blue]Saving data dictionary in {format} format...[/blue]")
        filepath = generator.save_dictionary(
            output_dir=output_dir,
            format=format,
            include_stats=include_stats
        )
        
        console.print(f"[green]✓ Data dictionary saved to {filepath}[/green]")
        
        # Display summary
        total_vars = len(generator.dictionary['variables'])
        by_source = {}
        by_category = {}
        
        for var_info in generator.dictionary['variables'].values():
            source = var_info['source']
            category = var_info['category']
            
            by_source[source] = by_source.get(source, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
        
        table = Table(title="Data Dictionary Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Total Variables", str(total_vars))
        table.add_row("Data Sources", str(len(generator.dictionary['data_sources'])))
        table.add_row("Variable Categories", str(len(generator.dictionary['variable_categories'])))
        
        for source, count in by_source.items():
            table.add_row(f"Variables from {source}", str(count))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error generating data dictionary: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info():
    """Display information about data dictionary generation."""
    console.print("[bold blue]Data Dictionary Generator Information[/bold blue]")
    
    console.print("\n[bold yellow]Supported Variable Types:[/bold yellow]")
    console.print("• US Treasury yield curves (11 tenors)")
    console.print("• Macroeconomic indicators from FRED")
    console.print("• European government bond yields")
    
    console.print("\n[bold yellow]Output Formats:[/bold yellow]")
    console.print("• Markdown (.md) - Human-readable documentation")
    console.print("• JSON (.json) - Machine-readable metadata")
    
    console.print("\n[bold yellow]Usage Examples:[/bold yellow]")
    console.print("• Generate markdown dictionary:")
    console.print("  [dim]python generate_data_dictionary.py generate[/dim]")
    console.print("• Generate JSON dictionary:")
    console.print("  [dim]python generate_data_dictionary.py generate --format json[/dim]")
    console.print("• Include statistical information:")
    console.print("  [dim]python generate_data_dictionary.py generate --include-stats[/dim]")


if __name__ == "__main__":
    app() 