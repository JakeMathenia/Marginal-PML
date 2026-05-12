# How to Use Jupyter Notebooks in this Project

## What are Jupyter Notebooks?

Jupyter notebooks (`.ipynb` files) are interactive documents that combine code, text, and visualizations. They're perfect for data analysis, prototyping, and documentation. Each notebook is made up of **cells** that can contain either code or markdown text.

## Available Notebooks in This Project

Located in the `notebooks/` directory:

1. **`01_data_ingestion.ipynb`** - Fetch NFIP claims data from OpenFEMA API
2. **`02_eda.ipynb`** - Exploratory Data Analysis with charts and statistics  
3. **`03_feature_engineering.ipynb`** - Transform raw claims into ML features
4. **`04_severity_model.ipynb`** - Train XGBoost severity prediction model
5. **`05_frequency_model.ipynb`** - Train XGBoost frequency prediction model
6. **`06_elt_generation.ipynb`** - Generate Event Loss Tables from ML models

## How to Run the Notebooks

### Option 1: Visual Studio Code (Recommended)
Since you're already using VS Code:

1. **Install the Python extension** (if not already installed)
2. **Click on any `.ipynb` file** in the file explorer
3. **VS Code will open it as a notebook** with interactive cells
4. **Select your Python interpreter**: 
   - Press `Ctrl+Shift+P` → Type "Python: Select Interpreter"
   - Choose the `.venv` interpreter from this project
5. **Run cells**: Click the ▶️ button next to each cell, or press `Shift+Enter`

### Option 2: Jupyter Lab (Web Interface)
If you prefer the classic Jupyter experience:

```bash
# Install Jupyter (may need to enable Windows long paths first)
pip install jupyterlab

# Launch Jupyter Lab
jupyter lab

# This opens a web browser at http://localhost:8888
# Navigate to the notebooks/ folder and click any .ipynb file
```

### Option 3: Use Google Colab
Upload the `.ipynb` files to [Google Colab](https://colab.research.google.com/) for a cloud-based experience.

## How to Use Each Notebook

### Getting Started
1. **Run cells in order** - notebooks are designed to be executed from top to bottom
2. **Each cell builds on previous cells** - variables and imports carry over
3. **Restart if needed**: If something breaks, use "Restart Kernel" to start fresh

### Key Cell Types
- **Code cells**: Contain Python code, run with `Shift+Enter`
- **Markdown cells**: Contain explanatory text, render with `Shift+Enter`
- **Output cells**: Show results, charts, tables automatically

### Example Workflow

**For `01_data_ingestion.ipynb`:**
```python
# Cell 1: Imports and setup
from src.data_ingestion import NFIPClient
import pandas as pd

# Cell 2: Configure API client  
client = NFIPClient()

# Cell 3: Fetch some sample data
claims = client.fetch_claims(
    state="FL", 
    year_of_loss=2022, 
    limit=1000
)

# Cell 4: Explore the data
print(f"Loaded {len(claims):,} claims")
claims.head()
```

**For `03_feature_engineering.ipynb`:**
```python
# Cell 1: Load the feature pipeline
from src.feature_engineering import FeaturePipeline

# Cell 2: Transform claims into ML features
pipeline = FeaturePipeline()
X, y_sev, y_freq = pipeline.fit_transform(claims)

# Cell 3: Examine the features
print(f"Feature matrix: {X.shape}")
X.describe()
```

## Tips for Success

### 1. **Set your working directory correctly**
Add this to the first cell of every notebook:
```python
import os
import sys
from pathlib import Path

# Ensure we're in the project root
os.chdir(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
sys.path.insert(0, "src")
```

### 2. **Handle missing data gracefully**
```python
# Check if data file exists before loading
data_path = Path("data/my_data.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
else:
    print(f"⚠️  {data_path} not found - running with synthetic data")
    df = make_synthetic_data()
```

### 3. **Save intermediate results**
```python
# Save processed data so you don't have to recompute
processed_claims.to_csv("data/processed_claims.csv", index=False)
```

### 4. **Use the validation script**
Before running the notebooks, ensure everything works:
```bash
python tests/test_pipeline.py
```

## Notebook vs. Production Code

- **Notebooks are for exploration** - experiment, visualize, iterate quickly
- **Python modules (`src/`) are for production** - clean, tested, reusable code
- **The notebooks call the src modules** - best of both worlds!

## Troubleshooting

### "Module not found" errors:
```python
# Add this at the top of your notebook
import sys
sys.path.insert(0, '../src')  # or 'src' if already in project root
```

### Kernel crashes or hangs:
- Click "Restart Kernel" and run cells again
- Check for infinite loops or memory issues

### VS Code not recognizing .ipynb files:
- Install the "Jupyter" extension from Microsoft
- Reload VS Code window (`Ctrl+Shift+P` → "Reload Window")

---

**Ready to start?** Open `notebooks/01_data_ingestion.ipynb` in VS Code and run the first few cells to see how it works!