# DAGPap24

This repo includes all content relevant to [DAGPap24 (Detection of Artificially Generated Scientific Papers 2024)](https://www.codabench.org/competitions/2431/#/pages-tab) competition, organised for SDP 2024

## Instructions

First, install the Python dependencies:

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

Second, run the command `python -m src.ml_gen_detection.dagpap24_baseline`.
## Data

- [Download training dataset](https://drive.google.com/file/d/1hJ-JtC0i8LBpD1hF3xWfRjkax42uE2NP/)
- [Download dev dataset](https://drive.google.com/file/d/1rurhsY7cbS1JoYtE4h2-vTVFUdMP8fFo/)
- [Download test dataset](https://drive.google.com/file/d/1Y7bALhTbsBZ-cza9QMlBu3_qgUeiFOdC/)

### Train data

The training set has the following columns:

- text – an excerpt from the article's full text;
- tokens – same as text, split by whitespaces;
- annotations – a list of triples [[start_id, end_id, label_id]] where ids indicate start and end token ids of a span, and label_id indicates its provenance ('human', 'NLTK_synonym_replacement', 'chatgpt', or 'summarized');
- token_label_ids – a list mapping each token from tokens with a corresponding label id (from 0 to 3), according to annotations.
```
>>> train_df = pd.read_parquet('train_data.parquet', engine='fastparquet')
>>> train_df[['text', 'tokens']].head(2)
	                                            text	                        annotations
index		
15096	Across the world, Emergency Departments are fa...	[[0, 3779, human], [3780, 7601, NLTK_synonym_r...
14428	lung Crab is the in the lead make of cancer-re...	[[0, 4166, NLTK_synonym_replacement], [4167, 2...


>>> train_df[["tokens", "token_label_ids"]].head(2)
	                                            tokens	                    token_label_ids
index		
15096	[Across, the, world,, Emergency, Departments, ...	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
14428	[lung, Crab, is, the, in, the, lead, make, of,...	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...

```

### Dev / Test data

The development and test sets have the following columns:

- text
- tokens
```
>>> dev_df = pd.read_parquet('dev_data.parquet', engine='fastparquet')
>>> dev_df.head()
                                                    text                                             tokens
index                                                                                                      
12313  Phylogenetic networks are a generalization of ...  [Phylogenetic, networks, are, a, generalizatio...
3172   Prediction modelling is more closely aligned w...  [Prediction, modelling, is, more, closely, ali...
6451   The heat transfer exhibits the flow of heat (t...  [The, heat, transfer, exhibits, the, flow, of,...
4351   a common experience during superficial ultraso...  [a, common, experience, during, superficial, u...
22694  Code metadata Current code version v1.5.9 Perm...  [Code, metadata, Current, code, version, v1.5....
```

## Evaluation

We're using Macro F1 score on `token_label_ids`. For each Full text, we're tokenizing/splitting the text on whitespace, and labeling each token. The final score is the average f1 across all full texts in the test set.

You can test your solution's performance offline by using the provided evaluation script [eval_f1.py](src/eval_f1.py)

Usage:
```
poetry run python -m src.eval_f1 --true_labels_file <file-in-data-dir-with-true-labels>.parquet --pred_file predictions.parquet
```

> Both files (true labels file and predictions file) must be located in [data](data)

