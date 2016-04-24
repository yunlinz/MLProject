import json
import nltk
import nltk.stem.snowball as sbstem
import business
import scipy.sparse as sp
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

DEBUG = False

def make_common_word_set():
    cw = []
    with open('common_words') as f:
        for l in iter(f):
            w = l.strip()
            cw.append(w)
    return set(cw)

def make_pronoun_set():
    pn = []
    with open('pronouns') as f:
        for l in iter(f):
            w = l.strip()
            pn.append(w)
    return set(pn)

cw_set = make_common_word_set()
pn_set = make_pronoun_set()
stemmer = sbstem.SnowballStemmer('english')
'''
Statis function for creating a bag of words data set based on a business file and a review file
'''
def create_bag_of_wods(businessfile=None, reviewfile=None, ba_aggr=None, attribute=None):
    # Creates a BusinessAggregator object and iterate through to add each business to the dataset
    if businessfile is not None and reviewfile is not None and ba_aggr is None:
        ba = business.BusinessAggregator(businessfile, reviewfile)
        ba_aggr = ba.aggr
    bag_of_words = BagOfWords()
    if attribute is None:
        bag_of_words.add_attribute('Caters')
    elif isinstance(attribute, str):
        bag_of_words.add_attribute(attribute)
    elif isinstance(attribute, list):
        for a in attribute:
            bag_of_words.add_attribute(a)
    else:
        raise Exception('Invalid attribute data type!!!')
    for bid, b in iter(ba):
        print 'Processing business Id: {}'.format(bid)
    for  b in ba_aggr.values():
        #print ('Processing business Id: {}'.format(bid))
        bag_of_words.add_datapoint(b)
    return bag_of_words

def create_bag_of_ngrams(businessfile, reviewfile, attribute=None, ngrams=3):
    # Creates a BusinessAggregator object and iterate through to add each business to the dataset
    ba = business.BusinessAggregator(businessfile, reviewfile)
    bag_of_ngrams = BagOfNgrams(n=ngrams)
    if attribute is None:
        bag_of_ngrams.add_attribute('Caters')
    elif isinstance(attribute, str):
        bag_of_ngrams.add_attribute(attribute)
    elif isinstance(attribute, list):
        for a in attribute:
            bag_of_ngrams.add_attribute(a)
    else:
        raise Exception('Invalid attribute data type!!!')
    for bid, b in iter(ba):
        print 'Processing business: {}'.format(bid)
        bag_of_ngrams.add_datapoint(b)
    return bag_of_ngrams


class DataSet:
    """
        DataSet abstract class that has functions that child classes can inherit
        """

    '''
        Initialize 4 fields for the class
        attributes: list of items in the attributes dict in the business metadata to look for (d attributes)
        labels: list of list of labels for the attributes dimension is n x d
        features: list of dict of features in sparse representation using dicts n x ???
        stars: list of list of star ratings from the reviews n x r (reviews)
        '''
    def __init__(self):
        self.attributes = []
        self.labels = []
        self.features_dict = []
        self.stars = []
        self.vocabulary = {}
        self.datamatrix = None
        self.isTfIdf = False


    def make_tfidf_matrix(self, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        try:
            self.datamatrix = TfidfTransformer(norm=norm,
                                               use_idf=use_idf,
                                               smooth_idf=smooth_idf,
                                               sublinear_tf=sublinear_tf)\
                .fit_transform(self.datamatrix)
            self.datamatrix.eliminate_zeros()
            self.isTfIdf = True
        except:
            print 'Tf-Idf Transformation failed!!!'

    '''
    Adds another attribute for the dataset builder to look for, will fail if we have already built a dataset
    '''
    def add_attribute(self, attribute):
        if not len(self.features_dict) == 0:
            print 'Already created dataset, cannot add more attributes!!!'
        else:
            if attribute not in self.attributes:
                self.attributes.append(attribute)

    '''
    Checks if the data is valid.
    '''
    def check_data(self):
        l_attributes = len(self.attributes)
        l_labels = len(self.labels)
        l_features = len(self.features_dict)
        l_stars = len(self.stars)
        if not l_attributes == l_labels or not l_labels == l_features or not l_features == l_stars:
            print Exception('Data lengths do not match!')
            return False
        return True

    def __str__(self):
        if not self.check_data():
            return 'Bad data!'
        output = ''
        output += str(self.attributes) + ':stars:features\n'
        for i in range(len(self.attributes)):
            output += str(self.labels) + ':' + str(self.stars) + ':' + str(self.features_dict) + '\n'
        return output

    '''
    Reads a business object and add as a datapoint. This will vary from one representation to the next, so this should
    be overridden in child implementations
    '''
    def add_datapoint(self, business):
        raise NotImplementedError('Please override in child class!!!')

    def make_sparse_datamtrix(self, mat_maker=sp.csr_matrix):
        indptr = [0]
        indices = []
        data = []
        for doc in self.features_dict:
            for feature, value in doc.iteritems():
                index = self.vocabulary.setdefault(feature, len(self.vocabulary))
                indices.append(index)
                data.append(value)
            indptr.append(len(indices))
        sparse_matrix = mat_maker((data, indices, indptr), dtype=float)
        if DEBUG:
            mat_rows = sparse_matrix.shape[0]
            mat_cols = sparse_matrix.shape[1]
            mat_size = mat_rows * mat_cols
            print 'Size of the sparse matrix is: {} x {}'.format(mat_rows, mat_cols)
            print 'Sparsity of data matrix is: {}'.format(float(sparse_matrix.nnz) / mat_size)
            if mat_size < 10000:
                print self.vocabulary
                print sparse_matrix.toarray()
            else:
                print sparse_matrix.toarray()[0:100, 0:100]
        self.datamatrix = sparse_matrix


class BagOfWords(DataSet):
    """
    Bag of words class
    Represents the data as dict of {word: count/total_words}
    count is occurrence in all reviews for that restaurant and total_words is total words for those reviews
    """
    def add_datapoint(self, business):
        this_attribute = []
        if not isinstance(business, type(business)):
            return
        # loops through the set of attributes that we should look for in this business and add it to the attribute list
        for a in self.attributes:
            attr = business.get_attribute(a)
            this_attribute.append(attr)
        self.labels.append(this_attribute)
        all_words = []
        all_stars = []
        # tokenize all reviews and add the star values to the respective lists
        for r in business.reviews:
            all_words += nltk.tokenize.word_tokenize(r['text'])
            all_stars.append(r['stars'])
        this_bag = {}
        self.stars.append(all_stars)
        total_words = 0
        # make bag of words
        for word in all_words:
            w = word.lower()
            if w in cw_set:
                continue
            total_words += 1
            if w in this_bag:
                this_bag[w] += 1
            else:
                this_bag[w] = 1
        # normalize bag of words by dividing by total words
        # for k in this_bag.iterkeys():
        #     this_bag[k] /= float(total_words)
        self.features_dict.append(this_bag)


class BagOfNgrams(DataSet):
    def __init__(self, n):
        DataSet.__init__(self)
        self.ngram = n

    def add_datapoint(self, business):
        this_attribute = []
        if not isinstance(business, type(business)):
            return
        # loops through the set of attributes that we should look for in this business and add it to the attribute list
        for a in self.attributes:
            attr = business.get_attribute(a)
            this_attribute.append(attr)
        self.labels.append(this_attribute)
        all_words = []
        all_stars = []
        # tokenize all reviews and add the star values to the respective lists
        for r in business.reviews:
            all_words += ngram_stemmed_review(r['text'], n=self.ngram)
            all_stars.append(r['stars'])
        this_bag = {}
        self.stars.append(all_stars)
        total_words = 0
        # make bag of words
        for word in all_words:
            if word in cw_set:
                continue
            total_words += 1
            if word in this_bag:
                this_bag[word] += 1
            else:
                this_bag[word] = 1
        # normalize bag of words by dividing by total words
        # for k in this_bag.iterkeys():
        #     this_bag[k] /= float(total_words)
        self.features_dict.append(this_bag)
'''
Some auxiliary methods
'''
def review_stemmer(text):
    '''
    Should be the same as ngram_stemmed_review(text, 1);
    :param text: text of the review as a single string
    :return: array of stemmed words
    '''
    return [create_general_token(stemmer.stem(word)) for word in simple_tokenizer(text)]

def ngram_stemmed_review(text, n=3):
    '''
    Creates an ngram set of stemmed words
    :param text:
    :return: list of all jgrams j<=n
    '''
    output = []
    stemmed_array = review_stemmer(text)

    output.append(stemmed_array)
    for j in range(1, n+1, 1):
        jgram = zip(*[stemmed_array[i:] for i in range(j)]) # create each j-gram array
        output.append(jgram)
    return [inner
            for outer in output
                for inner in outer]


def simple_tokenizer(text):
    '''
    The simple tokenizer that just tokenizes
    :param text: raw text in human readable format
    :return: list of words unaltered
    '''
    return nltk.tokenize.word_tokenize(text)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_pronoun(s):
    if s in pn_set:
        return True
    return False

def create_general_token(word):
    if is_number(word):
        return '#num#'
    elif is_pronoun(word):
        return '#prn#'
    else:
        return word