# MAGKT

Code for *MAGKT: Behavioral Transition Modeling with Memory Activation for Graph-Enhanced Knowledge Tracing*.

## Repository layout

```
configs/            model and training settings (default.yaml), dataset settings (datasets/), ablation variants (presets.yaml)
magkt/              data processing, graphs, model, memory, training and metrics
data/               datasets (not included, see data/README.md)
predictions/        test predictions and labels of the runs reported in Table 3
step1-step7         experiment scripts
verify_metrics.py   recomputes the metrics from predictions/
```

## Environment

Python 3.10 and PyTorch (CUDA 12.6). Install the PyTorch build that matches your CUDA driver
(https://pytorch.org), then `pip install -r requirements.txt`.

## Running the experiments

Each script has a settings block at the top and can be run directly (e.g. `python step4_main_experiments.py`).
Results are written to `results/`.

| step | file | purpose |
|---|---|---|
| 1 | `step1_prepare_data.py` | sequence files and hypergraph from the raw data |
| 2 | `step2_smoke_test.py` | quick test on synthetic data |
| 3 | `step3_tune.py` | selection of the dataset-specific settings on the validation learners (results already in `configs/datasets/`) |
| 4 | `step4_main_experiments.py` | MAGKT on the four datasets, seeds 0-4 |
| 5 | `step5_ablations.py` | ablation variants |
| 6 | `step6_time_decay.py` | lag measured in elapsed time |
| 7 | `step7_summarize.py` | collects the test metrics of all runs into `results/summary_long.csv` |

## Evaluation protocol

* Learner-level 80/20 train/test split; 10% of the training learners form the validation set.
* Sequences are cut into non-overlapping segments; memory and LSTM state are reset at every segment.
* Early stopping on the validation AUC (patience 10); the selected checkpoint is evaluated once on the test learners.
* Seeds 0-4. ACC and F1 use a threshold of 0.5 with the correct response as the positive class.

## Test predictions

`predictions/<dataset>/<model>/seed<k>.npz` contains the observed responses `y` and the predicted probabilities `p`
for all evaluated test interactions. `verify_metrics.py` recomputes AUC, ACC, RMSE, F1 and ECE from these files
and reports the mean and standard deviation over the seeds.
