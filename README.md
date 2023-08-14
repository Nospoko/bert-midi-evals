# BERT MIDI Evals

A set of Machine Learning tasks on top of the Maestro dataset. Designed to test BERT-based models pretrained on a larger Piano For AI dataset.

## Training
To create new model and train it you can run
```shell
python train.py
```
you can specify which composers you would like to classify by setting model.composers hyperparameter.

The program saves model to models/{path}, prints run_id, which you can later use to evaluate the run.
## Evaluating
To evaluate the model, run:
```shell
python eval.py run_id="run_id"
```
The program will print loss, accuracy and f1 score, that the model reached on test dataset.

## Dashboard
You can look through samples on which model performs good/bad. To start, run:
```shell
streamlit run --server.port 4466 dashboard.py
```

### Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
