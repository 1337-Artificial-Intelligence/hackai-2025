# hackai-challenges


# convert py file to ipynb file

```bash
jupytext --to notebook --output "hackai-challenges/new_notebooks/agents_prompt_engineering.ipynb" "hackai-challenges/py/agents_prompt_engineering.py"
```

# convert notebooks to python scripts
```bash
for f in hackai-challenges/Notebooks/*.ipynb; do jupytext --to py:percent "$f" --output "hackai-challenges/py/$(basename "$f" .ipynb).py"; done
```

