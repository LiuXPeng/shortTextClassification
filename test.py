from nltk.tokenize.stanford_segmenter import StanfordSegmenter
segmenter = StanfordSegmenter(path_to_jar="stanford-segmenter-3.4.1.jar", path_to_sihan_corpora_dict="./data", path_to_model="./data/pku.gz", path_to_dict="./data/dict-chris6.ser.gz")
sentence = 'l love you'
print(segmenter.segment(sentence))