import json
import nltk
import business

class DataSet:
    def __init__(self):
        self.attributes = []
        self.labels = []
        self.features = []
        self.stars = []


class BagOfWords(DataSet):
    def add_datapoint(self, business):
        this_attribute = []
        if not isinstance(business, business.Business):
            return
        for a in self.attributes:
            attr = business.get_attribute(a)
            this_attribute.append(attr)
        self.labels.append(this_attribute)
        all_words = []
        all_stars = []
        for r in business.reviews:
            all_words += nltk.tokenize.word_tokenize(r['text'])
            all_stars.append(r['stars'])
        this_bag = {}
        self.stars.append(all_stars)
        total_words = 0
        for word in all_words:
            total_words += 1
            if word in this_bag:
                this_bag[word] += 1
            else:
                this_bag[word] = 1
        for k in this_bag.iterkeys():
            this_bag[k] /= total_words
        self.features.append(this_bag)

    def add_attribute(self, attribute):
        if not len(self.features) == 0:
            print 'Already created dataset, cannot add more attributes!!!'
        else:
            if attribute not in self.attributes:
                self.attributes.append(attribute)

