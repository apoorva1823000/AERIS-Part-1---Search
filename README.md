# AERIS Medico Helper - Part 1

**CSE518 Artificial Intelligence - Monsoon 2026**

Part 1 is now submission- and deployment-ready.

## Course focus
- reactive-deliberative control
- deterministic problem formulation
- BFS / UCS / A*
- four heuristic functions
- admissibility
- consistency
- heuristic dominance
- deterministic replanning

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## External maps
The app supports:
- JSON maps
- CSV maps (`nodes.csv` + `edges.csv`, optional `requests.csv`)

Examples are included in `data/`.

## Faculty-style deliverables
- `app.py` - implementation
- `analysis/AERIS_Part1_Heuristic_Comparison.ipynb` - comparison notebook
- `analysis/heuristic_comparison_results.csv` - generated comparison results
- `report/AERIS_Part1_Progress_Report.pdf` - short progress report
- `data/` - non-hard-coded sample maps

## Streamlit deployment
1. Create GitHub repository: `aeris-part1-search`
2. Upload the contents of this folder to the repository root.
3. Open Streamlit Community Cloud.
4. Create an app from the repository.
5. Branch: `main`
6. Main file: `app.py`
7. Deploy.

Recommended public app name: `aeris-part1-search`.
