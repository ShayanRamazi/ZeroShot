import gzip
import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data_file="reviews_data.txt.gz"

with gzip.open ('reviews_data.txt.gz', 'rb') as f:
    for i,line in enumerate (f):
        print(line)
        break


def read_input(input_file):
    logging.info("reading file {0}...this may take a while".format(input_file))

    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            yield gensim.utils.simple_preprocess(line)



documents = list(read_input(data_file))
logging.info("Done reading data file")
model = gensim.models.Word2Vec (documents, size=150, window=10, min_count=2, workers=10)
model.train(documents,total_examples=len(documents),epochs=10)