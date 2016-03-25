import json

'''
Class for dealing with businesses
'''

'''
Creates a business object from string of the format
business_dict|review_dict|review_dict...
'''

def load_from_string(string):
    l = string.split('|')
    metadata = json.loads([0])
    b = Business(metadata)
    for review in l[1:]:
        b.add_review(json.loads(review))
    return b

'''
Creates a BusinessAggregator object from file with each line in the format of
business_dict|review_dict|review_dict...
'''
def load_dump(dumpfile):
    ba = BusinessAggregator()
    with open(dumpfile) as f:
        for line in iter(f):
            business = load_from_string(line)
            ba.add_business(business)
    return ba


'''
BusinessAggregator object:
fields:
    aggr = dict[business_id => Business object]
'''
class BusinessAggregator:
    def __init__(self):
        self.aggr = {}

    def __init__(self, restaurant_file, review_file):
        self.aggr = {}
        with open(restaurant_file) as rst:
            for line in iter(rst):
                jline = json.loads(line)
                b_id = jline['business_id']
                self.aggr[b_id] = Business(jline)
        with open(review_file) as rvw:
            for line in iter(rvw):
                jline = json.loads(line)
                b_id = jline['business_id']
                self.aggr[b_id].add_review(jline)

    def add_business(self, business):
        b_id = business.get_metadata('business_id')
        if not b_id in self.aggr:
            self.aggr[b_id] = business
        else:
            print '{} already in this list of businesses!'.format(b_id)

    def dump_all_business(self, outfile):
        with open(outfile, 'w') as o:
            for v in self.aggr.itervalues():
                o.write(str(v))

    def make_dataset_by_attribute(self, attribtue):
        for k, v in self.aggr.iteritems():
            if v.get_attribute(attribtue) is None:
                self.aggr.pop(k, None)

'''
Business object
fields:
    metadata[dict]: the business metadata from the business json
    reviews[list[dict]]: list of reviews for this business from review json
'''
class Business:
    def __init__(self):
        self.metadata = {}
        self.reviews = []

    def __init__(self, metadata):
        if not isinstance(metadata, dict):
            meta = json.loads(metadata)
        else:
            meta = metadata
        self.metadata = meta
        self.reviews = []

    def get_metadata(self, field):
        return self.metadata[field]

    def add_review(self, review):
        if not isinstance(review, dict):
            rvw = json.loads(review)
        else:
            rvw = review
        self.reviews.append(rvw)

    def get_attribute(self, attribute):
        self.metadata['attributes'].fetch(attribute, None)

    def __iter__(self):
        return iter(self.reviews)

    def __str__(self):
        return json.dumps(self.metadata) + '|' + '|'.join(map(json.dumps, self.reviews)) + '\n'
