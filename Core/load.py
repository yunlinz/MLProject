import os
import json


def split_business_by_state(infile, outfile=None):
    if outfile is None:
        outfile = infile
    outfiles = {}
    with open(infile) as f:
        for line in iter(f):
            jline = json.loads(line)
            state = jline['state']
            if state not in outfiles:
                print 'adding new state {}'.format(state)
                o = outfile + '_' + state
                outfiles[state] = open(o, 'w')
            outfiles[state].write(line)
    for v in outfiles.itervalues():
        v.close()


def get_reviews_for_state(businesses, reviews, outfile=None):
    if outfile is None:
        outfile = businesses + '_reviews'
    b_ids = []
    with open(businesses) as f:
        for line in iter(f):
            jline = json.loads(line)
            b_ids.append(jline['business_id'])
        b_set = set(b_ids)

    with open(reviews) as f:
        with open(outfile, 'w') as o:
            for line in iter(f):
                jline = json.loads(line)
                b_id = jline['business_id']
                if b_id in b_set:
                    o.write(line)


def get_attributes(business_file, outfile=None):
    if outfile is None:
        outfile = business_file + '_attributes'
    attr_hash = {}
    with open(business_file) as f:
        for line in iter(f):
            jline = json.loads(line)
            attributes = jline['attributes']
            for k, v in attributes.iteritems():
                if k not in attr_hash:
                    attr_hash[k] = []
                values = []
                if not isinstance(v, dict):
                    values.append(v)
                else:
                    for val in v.iterkeys():
                        values.append(val)
                for value in values:
                    if value not in attr_hash[k]:
                        attr_hash[k].append(value)
    with open(outfile, 'w') as o:
        for k, v in attr_hash.iteritems():
            o.write(k + str(v) + '\n')


def get_restaurants(business_file, outfile=None):
    if outfile is None:
        outfile = business_file + '_restaurants'
    with open(outfile, 'w') as o:
        with open(business_file, 'r') as f:
            for line in iter(f):
                jline = json.loads(line)
                if 'Restaurants' in jline['categories']:
                    o.write(line)


if __name__ == '__main__':
    data_dir = '../data/'
    parsed_dir = data_dir + 'parsed/'
    raw_dir = data_dir + 'yelp_data/'

    raw_reviews = raw_dir + 'yelp_academic_dataset_review.json'
    business_data = raw_dir + 'yelp_academic_dataset_business.json'
    split_business_by_state(business_data, outfile=parsed_dir + 'businesses')
    get_restaurants('../data/parsed/businesses_WI')
    get_reviews_for_state(parsed_dir + 'businesses_WI_restaurants', raw_reviews)
    get_attributes('../data/yelp_data/yelp_academic_dataset_business.json', '../data/parsed/attributes')

    # get_reviews_for_state('../data/parsed/businesses_TX', '../data/yelp_data/yelp_academic_dataset_review.json')