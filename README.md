# Modeling Dual-Serotype Dengue Dynamics Under Climatic and Entomological Forcing

This repository contains the source code and datasets required to reproduce the results presented in the manuscript: 
**"Modeling Dual-Serotype Dengue Dynamics Under Climatic and Entomological Forcing: Insights from Southern Brazil"** (Ref: rsif-2026-0198), submitted to the *Journal of the Royal Society Interface*.

## Overview

The provided Python code implements a discrete-time stochastic compartmental model to simulate the transmission dynamics of two dengue serotypes. The model incorporates:
- **Climatic Forcing:** Daily temperature variations affecting transmission rates.
- **Entomological Forcing:** Premise Infestation Index (IIP) as a proxy for mosquito density.
- **Biological Complexity:** Antibody-Dependent Enhancement (ADE), temporary cross-immunity, and branching pathways for symptomatic ($I$) and asymptomatic ($A$) infections.
- **Numerical Stability:** The stochastic engine ensures mass conservation with a precision error (RMSE) of $10^{-13}$ relative to the total population $N$.

## Repository Structure

- `ModelTwoStrainsStochastic.py`: Main simulation script.
- `/data`:
    - `Clima_Foz.csv`: Historical temperature data for Foz do Iguaçu.
    - `Ento_Foz.csv`: Entomological surveillance data (IIP).
    - `Casos_Foz.csv`: Epidemiological data (reported dengue cases).
- `requirements.txt`: List of required Python dependencies.

## Requirements

The code is written in **Python 3.8+**. The following libraries are required:
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

You can install the dependencies using:
```bash
pip install -r requirements.txt
