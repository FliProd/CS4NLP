
import matplotlib.pyplot as plt


def explore(df):

    print('Number of sentence versions per dialect')
    print(df.groupby('dialect')['sentence_version'].count())

    df['sentence_char_count'] = df['sentence_version'].apply(lambda x: len(x))
    print('Average number of chars per dialect')
    print(df.groupby('dialect').mean()['sentence_char_count'])

    df['num_words'] = df['sentence_version'].apply(lambda x: x.count(' '))
    print('Average number of words per dialect')
    print(df.groupby('dialect').mean()['num_words'])

    df['avg_char_count_per_word'] = df['sentence_version'].apply(avg_char_count_per_word)
    print('Average char count per word per dialect')
    print(df.groupby('dialect').mean()['avg_char_count_per_word'])

    # histogram of word frequencies over all dialects
    plot_histogram_of_word_frequencies(df)

    


def plot_histogram_of_word_frequencies(df):
    df['sentence_version'] = df['sentence_version'].apply(lambda x: x.split(' '))
    df = df.explode('sentence_version').rename(columns={'sentence_version': 'words'})[['dialect', 'words']]
    df = df.groupby('words').size().sort_values(ascending=False).reset_index(name="count")
    df['count'].hist(bins=200, log=True)
    plt.show()


def avg_char_count_per_word(sentence):
    words = sentence.split(' ')
    return len(sentence.replace(' ', '')) / len(words)
