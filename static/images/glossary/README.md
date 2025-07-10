# Financial Glossary Visual Aids

This directory contains visual aids and icons for the YieldCurveAI Financial Glossary.

## Current Visual Aids

### Placeholder Images
The following visual aids are referenced in the glossary but not yet implemented:

- `yield_curve_example.png` - Example yield curve chart showing normal, inverted, and flat curves
- `tenor_timeline.png` - Timeline visualization showing different Treasury maturity periods
- `fed_funds_chart.png` - Historical Fed Funds Rate chart with policy periods

### Future Enhancements

Consider adding the following visual aids:

1. **Concept Diagrams:**
   - Yield curve shapes (normal, inverted, flat, humped)
   - Model architecture diagrams
   - Feature importance visualizations

2. **Icons:**
   - Category icons for each glossary section
   - Model type icons
   - Metric visualization icons

3. **Interactive Elements:**
   - Animated yield curve movements
   - Model prediction confidence bands
   - Economic indicator correlations

## Usage

Visual aids are automatically loaded by the glossary page when:
1. The `visual` field is specified in `config/glossary.yaml`
2. The corresponding image file exists in this directory
3. The file path is properly referenced

## Supported Formats

- PNG (recommended for charts and diagrams)
- JPG (for photographs and complex images)
- SVG (for scalable icons and simple graphics)

## File Naming Convention

Use descriptive names that match the glossary term:
- `term_name_example.png` for example visualizations
- `term_name_chart.png` for chart-based explanations
- `term_name_icon.svg` for simple icons
- `term_name_diagram.png` for conceptual diagrams 