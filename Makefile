PYTHON ?= python3

.PHONY: check tables figures paper reproduce-oracle reproduce-sensitivity reproduce-crossfit

check:
	$(PYTHON) code/verify_results.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) -m unittest discover -s tests -v

tables:
	$(PYTHON) code/gen_appendix.py

figures:
	$(PYTHON) code/make_figures.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex

reproduce-oracle:
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) code/oracle_sim.py
	OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $(PYTHON) code/quadrature.py

reproduce-sensitivity:
	$(PYTHON) code/sensitivity_check.py --workers 2

reproduce-crossfit:
	$(PYTHON) code/run_studies.py --workers 2
