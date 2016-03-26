import json
import nltk
import business

DEBUG = True

'''
Statis function for creating a bag of words data set based on a business file and a review file
'''
def create_bag_of_wods(businessfile, reviewfile, attribute=None):
    # Creates a BusinessAggregator object and iterate through to add each business to the dataset
    ba = business.BusinessAggregator(businessfile, reviewfile)
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
    for b in iter(ba):
        bag_of_words.add_datapoint(b)
    if DEBUG:
        print str(bag_of_words)
    return bag_of_words


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
        self.features = []
        self.stars = []

    '''
    Adds another attribute for the dataset builder to look for, will fail if we have already built a dataset
    '''
    def add_attribute(self, attribute):
        if not len(self.features) == 0:
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
        l_features = len(self.features)
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
            output += str(self.labels) + ':' + str(self.stars) + ':' + str(self.features) + '\n'
        return output

    '''
    Reads a business object and add as a datapoint. This will vary from one representation to the next, so this should
    be overridden in child implementations
    '''
    def add_datapoint(self, business):
        raise NotImplementedError('Please override in child class!!!')



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
            total_words += 1
            if word in this_bag:
                this_bag[word] += 1
            else:
                this_bag[word] = 1
        # normalize bag of words by dividing by total words
        for k in this_bag.iterkeys():
            this_bag[k] /= float(total_words)
        self.features.append(this_bag)

