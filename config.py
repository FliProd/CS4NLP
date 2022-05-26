config = dict()

"""
Data settings
"""

config['numbers'] = False
config['numbers_dataset'] = 'sentences_ch_de_numerics'
config['transcribed_dataset'] = 'sentences_ch_de_transcribed'
config['raw_data_path'] = 'data/raw/'
config['processed_data_path'] = 'data/processed/'
config['n'] = [1, 2, 3, 4]
config['dialects'] = ['ch_sg', 'ch_be', 'ch_gr', 'ch_zh', 'ch_vs', 'ch_bs', 'ch_ag', 'ch_lu']



"""
Preprocessing settings
Note that (,),[,],& and other symbols only appear <10 times and are not worth the computational complexity
"""


config['symbols_to_remove'] = ['"', '«', '»', '.', ',', '?', '!', '(', ')', '[', ']', '&']

config['stopwords_threshold_total'] = 200
config['stopwords_threshold_tf_idf'] = 0.09
