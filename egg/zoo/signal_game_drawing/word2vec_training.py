from datasets import load_dataset
from gensim.models import Word2Vec


dataset = load_dataset('ag_news')
sentences = []
for split in dataset.values():  # Iterate through train/test splits
    for example in split:
        # Tokenize the text (simple split by whitespace)
        tokens = example['text'].lower().split()
        sentences.append(tokens)

word2vec = Word2Vec(sentences=sentences, vector_size=512, window=5, min_count=1, workers=4)

word2vec.save('/home/casper/Documents/Github/EGG/egg/zoo/signal_game_drawing/data/word2vec.model')

cate_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']