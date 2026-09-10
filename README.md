# Efficient estimation and the cost of complete-case coarsening under monotone sequential MAR

**Keivan Bolouri · University of California, Irvine**

Research code and simulation results for estimating an average treatment effect when two confounders are partially observed in a monotone pattern. The manuscript studies the information gained by using partially complete records, and the bias that can arise when those records are reduced to a complete-case indicator.

**[Read the paper](paper/paper.pdf)** · **[LaTeX source](paper/paper.tex)** · **[Reproduction guide](docs/REPRODUCIBILITY.md)**

## Start here

| Contents | Location |
| --- | --- |
| Manuscript, LaTeX inputs, and figures | [paper/](paper/) |
| Simulation programs and table/figure generators | [code/](code/) |
| Results, configurations, and replicate-level CSV files | [results/](results/) |
| Implementation checks | [tests/](tests/) |
| Instructions, data definitions, and validation | [docs/](docs/) |

All study data are synthetic. The current results include 179,500 estimator records from 48,500 simulated samples. The code and results at these locations correspond to the current manuscript.

## Verify the supplied results

Use Python 3.12 and the recorded package versions:

```bash
python -m pip install -r requirements.txt
python code/verify_results.py
python -m unittest discover -s tests -v
```

The [reproduction guide](docs/REPRODUCIBILITY.md) gives environment setup, all simulation commands, random seeds, and output definitions. The [run manifest](results/run_manifest.json) records every estimated-nuisance configuration.

## Build the manuscript

With a standard LaTeX installation, run:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
```

The supplied figures and numerical sections are ready to compile. Keep `paper.tex`, `simulation.tex`, `appendix_d.tex`, and `figures/` together. References are included in the source; BibTeX is unnecessary.

## Interpretation and citation

The paper separates efficiency under common validity from identification when complete-case MAR fails. Estimated-nuisance results also show undercoverage in the Gaussian stress design and do not establish a uniform finite-sample advantage.

Citation metadata are in [CITATION.cff](CITATION.cff). The original CCMAR software associated with Levis and colleagues is available in [flex-ate-confounders-MAR](https://github.com/alexlevis/flex-ate-confounders-MAR).

Earlier code and results are retained in [archive/previous_version/](archive/previous_version/) for provenance. They belong to earlier analyses; use the main folders above to reproduce the current manuscript.
