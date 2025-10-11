# WAT 2025 English to Indic Multimodal Translation

This repository contains code for the WAT 2025 English to Indic Multimodal Translation task.

## Setup

This project uses `uv` for dependency management and `ruff` for linting and formatting.

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd wat-2025-english2indic-mmt
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
# Option 1: Use the activation script
source activate_env.sh

# Option 2: Manual activation
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Automatic Terminal Activation

This project is configured to automatically activate the virtual environment when you open a new terminal in Cursor:

1. **Open the workspace file**: Open `wat-2025-english2indic-mmt.code-workspace` in Cursor
2. **New terminals will auto-activate**: Every new terminal you create will automatically activate the virtual environment
3. **Fallback activation**: If auto-activation doesn't work, use `source activate_env.sh`

### Development

#### Linting and Formatting

This project uses `ruff` for both linting and formatting:

```bash
# Check for linting issues
uv run ruff check .

# Fix auto-fixable linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Run both linting and formatting
uv run ruff check . --fix && uv run ruff format .
```

#### Testing

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov
```

## Project Structure

```
├── data/                    # Dataset files
│   ├── bengali/            # Bengali dataset
│   └── hindi/              # Hindi dataset
├── src/                    # Source code (to be created)
├── tests/                  # Test files (to be created)
├── pyproject.toml          # Project configuration
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Data

The dataset contains:
- Bengali Visual Genome data (train/dev/test/challenge sets)
- Hindi Visual Genome data (train/dev/test/challenge sets)

Each language has:
- Training set: ~29K images
- Development set: ~1K images  
- Test set: ~1.6K images
- Challenge test set: 1.4K images
