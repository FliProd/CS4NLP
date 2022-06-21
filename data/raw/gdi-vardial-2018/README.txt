== Data Format ==

The training data contains the following files:

	train.txt - training set
	dev.txt   - development/validation set

Each line in the *.txt files is tab-delimited in the format:

	sentence<tab>dialect-label

You can use the development set for model selection to avoid overfitting the training set. This year's training set is an updated and expanded version of the 2017 training set, and this year's development set roughly corresponds to the 2017 test set. The training data contains utterances from four Swiss German dialects: Bern (BE), Basel (BS), Lucerne (LU) and Zurich (ZH). The training set contains data from 3-7 speakers per dialect, and the development set contains data from 1 speaker per dialect. Due to this rather small number of speakers, we recommend you to train your final chosen system on the combined training and development sets.

=== About the transcriptions ===

There is no widely spread convention for writing Swiss German. All instances have been transcribed using the writing system "Schwyzertütschi Dialäktschrift" proposed by Dieth (1986) to provide some guidance on how to write in a Swiss German dialect. The transcription is expected to show the phonetic properties of the variety but in a way that is legible for everybody who is familiar with standard German orthography (Dieth, 1986, 10). Dieth's system, which is originally phonemic, can be implemented in different ways depending on how differentiated the phonetic qualities are to be expressed. Although it is the objective to keep track of the pronunciation, Dieth's transcription method is orthographic and partially adapted to spelling habits in standard German. Therefore it does not provide the same precision and explicitness as phonetic transcription methods do. The transcriptions exclusively use lower case characters. Note that Dieth's system is hardly known by laymen, so that e.g. Swiss German data extracted from social media may look fairly different from our transcriptions.

In our transcriptions, we do not use the full power of phonemic distinctions available in the Dieth script. The grapheme inventory in the Dieth script is always related to the dialect and its phonetic properties, so that, for example, the grapheme <e> stands for different vowel qualities, [e], [ɛ] or [ə], depending on the dialect, the accentuation of the syllable and - to substantial degree - also to the dialectal background of the transcriber. For each dialect area, we include instances of several speakers and several transcribers.

== Evaluation ==

The test data (to be released later) will only contain sentences without their dialect labels.

Participants will be required to submit the labels for these test instances.

The exact details of the submission file format will be provided later.
