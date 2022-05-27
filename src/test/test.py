from config import config
from nltk import ngrams

def get_n_grams(word, n):
    ngrams_list = list(ngrams(' ' + word + ' ', n))
    n_grams = [''.join(x) for x in ngrams_list]
    return n_grams

def get_dg_tn(n_grams, n, dialect_model):
    nstring = f'{n}_grams'
    dg = 0.0
    for n_gram in n_grams:
        if n_gram in dialect_model[nstring]:
            dg += 1.0
    return dg

def get_vg(word, n, dialect_model):
    nstring = f'{n}_grams'
    n_grams = get_n_grams(word, n)
    dg_tn = get_dg_tn(n_grams, n, dialect_model)
    if dg_tn == 0:
        return config["penalty_p"]
    sum = 0.0
    for n_gram in n_grams:
        if n_gram in dialect_model[nstring]:
            sum += dialect_model[nstring][n_gram]
    return sum / dg_tn


def predict_dialect(df, n, models):
    scores = {}
    for dialect in config['dialects']:
        scores[dialect] = 0.0
        for word in df['sentence_version'].split(" "):
            scores[dialect] += get_vg(word, n, models[dialect])
    return (min(scores, key=scores.get))

def evaluate(data, predictions):
    this_dialects = {}
    for dialect in config['dialects']:
        this_dialects[dialect] = {
            "true_positives": 0,
            "positives": 0,
            "false_positives": 0,
        }
    for index, row in data.iterrows():
        actual = row["dialect"]
        predicted = predictions.loc[index]
        this_dialects[actual]["positives"] += 1
        if actual == predicted:
            this_dialects[actual]["true_positives"] += 1
        else:
            this_dialects[predicted]["false_positives"] += 1
    print("{:<10} {:>10} {:>10} {:>10}".format("", "Precision", "Recall", "F1-score"))
    for key, elem in this_dialects.items():
        elem["precision"] = elem["true_positives"] / (elem["true_positives"] + elem["false_positives"])
        elem["recall"] = elem["true_positives"] / elem["positives"]
        elem["f1"] = 2.0 * (elem["precision"] * elem["recall"]) / (elem["precision"] + elem["recall"])

        print("{:<9}:     {:<.4f}     {:<.4f}     {:<.4f}".format(key , elem["precision"], elem["recall"], elem["f1"]))

    total_precision = sum([x["precision"] for x in this_dialects.values()])/len(config['dialects'])
    total_recall = sum([x["recall"] for x in this_dialects.values()])/len(config['dialects'])
    total_f1 = sum([x["f1"] for x in this_dialects.values()])/len(config['dialects'])
    print("\n{:<9}:     {:<.4f}     {:<.4f}     {:<.4f}".format("Total" , total_precision, total_recall, total_f1))

def test(data, models):
    predictions = data.apply(lambda x: predict_dialect(x, 4, models), axis=1)
    evaluate(data, predictions)