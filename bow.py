from random import seed, shuffle

seed(123)

vocab_file = 'vocab.stof.txt'
train_file = 'train.stof.txt'
valid_file = 'valid.stof.txt'
test_file = 'test.stof.txt'

# read vocab file
vocab = list()
fp = open(vocab_file, 'r', encoding='utf-8')
for line in fp:
    word = line.strip()
    vocab.append(word)
fp.close()

# read docword file
def read_file(file):
    docword = list()
    fp = open(file, 'r')
    prev_doc_id = 0
    doc = dict()
    for line in fp:
        doc_id, vocab_id, count = line.strip().split()
        if doc_id != prev_doc_id:
            if doc:
                docword.append(doc)
                doc = dict()
        doc[int(vocab_id)] = int(count) 
        prev_doc_id = doc_id
    docword.append(doc)
    fp.close()
    return docword

# permute data randomly
W = len(vocab)
train = read_file(train_file)
D = len(train)
valid = read_file(valid_file)
test = read_file(test_file)

shuffle(train)
"""
for doc in train:
    for vocab_id in doc:
        print(vocab[vocab_id], doc[vocab_id])
    print('----')
"""
