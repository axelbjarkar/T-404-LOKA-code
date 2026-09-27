# T-404-LOKA-code

To setup:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pre-commit install
python -m ipykernel install --user --name=loka --display-name "Python (LOKA)"
```

To use the `loka` package on other machines / cloud notebooks (e.g. Google Colab) without cloning the repo:
```
pip install git+https://github.com/axelbjarkar/T-404-LOKA-code.git
```
