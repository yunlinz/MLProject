import json

def load_from_string(string):
    l = string.split('|')
    metadata = json.loads([0])
    b = Business(metadata)
    for review in l[1:]:
        b.add_review(json.loads(review))
    return b


def bucket_reviews(restaurants, reviews):
    aggr = {}
    with open(restaurants) as rst:
        for line in iter(rst):
            jline = json.loads(line)
            b_id = jline['business_id']
            aggr[b_id] = Business(jline)
    with open(reviews) as rvw:
        for line in iter(rvw):
            jline = json.loads(line)
            b_id = jline['business_id']
            aggr[b_id].add_review(jline)
    return aggr


def dump_all_businesses(aggr_dict, outfile):
    with open(outfile, 'w') as o:
        for v in aggr_dict.itervalues():
            o.write(str(v))


class Business:
    def __init__(self, metadata):
        if not isinstance(metadata, dict):
            meta = json.loads(metadata)
        else:
            meta = metadata
        self.metadata = meta
        self.reviews = []

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
        #TODO create to string method!
        return json.dumps(self.metadata) + '|' + '|'.join(map(json.dumps, self.reviews)) + '\n'
