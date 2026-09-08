PYTHON ?= python3

.PHONY: check figures paper reproduce-main reproduce-sensitivity reproduce-crossfit

check:
	$(PYTHON) -m py_compile code/*.py
	$(PYTHON) code/check_repository.py

figures:
	$(PYTHON) code/make_manuscript_figures.py

paper: figures
	latexmk -xelatex -interaction=nonstopmode -halt-on-error paper.tex

reproduce-main:
	$(PYTHON) code/validate_sequential_mar.py --n 2500 --reps 2000 --seed 20260831 --outdir scratch_results/main
	$(PYTHON) code/summarize_verified_simulation.py --replicates scratch_results/main/simulation_replicates.csv --truth scratch_results/main/truth.csv --outdir scratch_results/main --bootstrap-reps 10000 --seed 20260904
	$(PYTHON) code/verify_variance_identity.py --validator-dir code --draws 1500000 --sample-size 2500 --seed 86420 --out scratch_results/main/theoretical_variance_check.csv

reproduce-sensitivity:
	$(PYTHON) code/oracle_sensitivity_simulation.py --validator-dir code --n 2500 --reps 5000 --seed 20260904 --outdir scratch_results/oracle_sensitivity

reproduce-crossfit:
	$(PYTHON) code/crossfit_sequential_ate.py --n 4000 --reps 100 --folds 2 --seed 20260904 --outdir scratch_results/crossfit
