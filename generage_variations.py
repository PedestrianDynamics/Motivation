import json
import logging
import pathlib
from typing import Any, Dict, List, Union
import typer
from datetime import datetime

def get_nested_value(config: Dict, param_path: str) -> Any:
    """Get value from nested dictionary using dot notation."""
    keys = param_path.split('/')
    current = config
    for key in keys:
        current = current[key]
    return current

def set_nested_value(config: Dict, param_path: str, value: Any) -> None:
    """Set value in nested dictionary using dot notation."""
    keys = param_path.split('/')
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value

def create_parameter_variations(
    base_config: Dict,
    param_path: str,
    values: List[Union[float, int, str]],
    description: str = ""
) -> List[Dict]:
    """
    Create variations of the configuration by varying a single parameter.
    
    Args:
        base_config: The base configuration dictionary
        param_path: Path to the parameter using '/' notation (e.g., 'motivation_parameters/height')
        values: List of values to try for this parameter
        description: Optional description of what these variations represent
    
    Returns:
        List of variation dictionaries
    """
    original_value = get_nested_value(base_config, param_path)
    variations = []
    
    for value in values:
        variation = {
            "name": f"{param_path.replace('/', '_')}_{value}",
            "description": description,
            "original_value": original_value,
            "parameters": {
                param_path: value
            }
        }
        variations.append(variation)
    
    return variations

def save_variations(variations: List[Dict], output_path: pathlib.Path) -> None:
    """Save variations to a JSON file with proper formatting."""
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(variations, f, indent=4)

def main(
    inifile: pathlib.Path = typer.Option(
        pathlib.Path("files/inifile.json"),
        help="Path to the initial configuration file",
    ),
    param: str = typer.Option(
        None,
        help="Parameter to vary (using '/' notation, e.g., 'motivation_parameters/height')",
    ),
    values: str = typer.Option(
        None,
        help="Comma-separated list of values to try (e.g., '0.5,1.0,1.5')",
    ),
    description: str = typer.Option(
        "",
        help="Optional description of these variations",
    ),
    output: pathlib.Path = typer.Option(
        None,
        help="Output path for variations file (default: variations_PARAMETER_TIMESTAMP.json)",
    ),
) -> None:
    """Generate parameter variations for simulation runs."""
    if not param or not values:
        print("Usage example:")
        print("  python generate_variations.py --inifile files/inifile.json \\\n"
              "         --param motivation_parameters/height \\\n"
              "         --values 0.5,1.0,1.5 \\\n"
              "         --description 'Testing different heights'")
        return

    # Load base configuration
    with open(inifile, "r", encoding="utf8") as f:
        base_config = json.load(f)
    
    # Parse values (handle both float and int)
    try:
        parsed_values = []
        for v in values.split(','):
            v = v.strip()
            if '.' in v:
                parsed_values.append(float(v))
            else:
                parsed_values.append(int(v))
    except ValueError as e:
        print(f"Error parsing values: {e}")
        return

    # Generate variations
    variations = create_parameter_variations(
        base_config,
        param,
        parsed_values,
        description
    )
    
    # Determine output path
    if output is None:
        param_name = param.replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = pathlib.Path(f"variations_{param_name}_{timestamp}.json")
    
    # Save variations
    save_variations(variations, output)
    print(f"Created {len(variations)} variations in {output}")
    print("\nVariations summary:")
    for var in variations:
        print(f"- {var['name']}: {var['parameters'][param]}")

if __name__ == "__main__":
    typer.run(main)
