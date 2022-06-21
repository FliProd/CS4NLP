from nltk import ngrams
import pandas as pd

"""
Gets word an returns all n-grams of size n in it.
"""
def get_n_grams(word: str, n: int) -> list:
    ngrams_list = list(ngrams(' ' + word + ' ', n))
    n_grams = []
    for word_n_grams in ngrams_list:
        word_n_grams_joined = ''.join(word_n_grams)
        if word_n_grams_joined == " " or word_n_grams_joined == "  ":
            continue
        n_grams.append(word_n_grams_joined)
    return n_grams

"""
Gets sentence and returns all n-grams of size n in it.
"""
def get_n_grams_sentence(sentence:str, n:int) -> list:
    n_grams = []
    for i in sentence.split():
        n_grams += get_n_grams(i, n)
    return n_grams

"""
Gets a sentence and a list of symbols and removes all of them in it.
"""
def remove_from_sentence(sentence:str, symbols:list) -> str:
    for symbol in symbols:
        sentence = sentence.replace(symbol, '')
    return sentence


"""
Gets true_labels, predictions and dialects and evaluates them:
prints confusion matrix and precision, recall, f1, n_elems for each dialect
"""
def evaluate(true_labels: pd.DataFrame, predictions: pd.DataFrame, dialects: list) -> None:
    this_dialects = {}
    for dialect in dialects:
        this_dialects[dialect] = {
            "true_positives": 0,
            "positives": 0,
            "false_positives": 0,
        }
        for other_dialect in dialects:
            this_dialects[dialect][other_dialect] = 0
    
    for index, row in true_labels.iterrows():
        actual = row["dialect"]
        predicted = predictions.loc[index]
        this_dialects[actual]["positives"] += 1
        this_dialects[actual][predicted] += 1
        if actual == predicted:
            this_dialects[actual]["true_positives"] += 1
        else:
            this_dialects[predicted]["false_positives"] += 1

    print("\nConfusion matrix:\n      x-axis: actual values\n      y-axis: predicted values\n")
    format_row = "{:<6} " * (len(dialects) + 1)
    format_row = format_row
    print(format_row.format("", *[x[3:].upper() for x in dialects]))
    for dialect in dialects:
        print(format_row.format(dialect[3:].upper(), *[str(value[dialect]) for value in this_dialects.values()]))

    print("\n-------------------------------\n")
    print("{:<10} {:>10} {:>10} {:>10} {:>10}".format("", "Precision", "Recall", "F1-score", "n_elems"))
    for key, elem in this_dialects.items():
        elem["precision"] = elem["true_positives"] / (elem["true_positives"] + elem["false_positives"])
        elem["recall"] = elem["true_positives"] / elem["positives"]
        elem["f1"] = 2.0 * (elem["precision"] * elem["recall"]) / (elem["precision"] + elem["recall"])

        print("{:<9}     {:<.4f}     {:<.4f}     {:<.4f} {:>10}".format(key[3:].upper() , elem["precision"], elem["recall"], elem["f1"], elem["positives"]))

    total_precision = sum([x["precision"] for x in this_dialects.values()])/len(dialects)
    total_recall = sum([x["recall"] for x in this_dialects.values()])/len(dialects)
    total_f1 = sum([x["f1"] for x in this_dialects.values()])/len(dialects)
    total_elems = sum([x["positives"] for x in this_dialects.values()])
    print("\n{:<9}     {:<.4f}     {:<.4f}     {:<.4f} {:>10}".format("Average" , total_precision, total_recall, total_f1, total_elems))

