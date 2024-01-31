# This Case Study repository includes all necessary files for the given scenario
**topic_grouping.py**: Groups 'texts' by topic in the 'opinions.csv' file depending on the content utilizing 'sentence-transformers' library and 'cosine similarity'.
**opinion_classification.py**: Classifies 'texts' in the 'opinions.csv' file depending on the content similarity with the given labels ['claim', 'counterclaim', 'evidence', 'rebuttal'] and stores the results into a csv file, utilizing 'Zero-Shot Classification' models from huggingface platform.
**evaluation_type.py**: Evaluates the results of the 'opinion_pred.csv' file that contains predictions. (opinion type prediction)
