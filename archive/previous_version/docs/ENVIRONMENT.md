# Verification environment

The archived Python results were checked with:

- Python 3.12.13
- NumPy 2.3.5
- pandas 2.2.3
- SciPy 1.17.0
- Matplotlib 3.10.8
- scikit-learn 1.8.0

The manuscript was compiled with:

- XeTeX 3.141592653-2.6-0.999995
- TeX Live 2023/Debian
- Latexmk 4.83

The exact Python versions are recorded in requirements-lock.txt. The more
flexible lower bounds in requirements.txt are intended for ordinary use.

R was unavailable in the verification environment. Therefore, the Python
implementation and its stored outputs are authoritative; the R file is
included as a mirror/reference implementation.
