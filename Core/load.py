import os
import json
from business import Business
from sklearn.cross_validation import train_test_split
import modeldata as md
import mlmodels as mlm

'''
Takes as input a file with a business info on each line and splits it into separate files for each state
'''
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


'''
Takes a file with a set of businesses and reviews, save a set of reviews that only relevant to the businesses
'''
def get_reviews_for_businesses(businesses, reviews, outfile=None):
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

'''
Based on a business file, output a file with all possible attributes and values
'''
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

'''
Reads a business file and get only restaurants
'''
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
    '''
    some setting up
    '''
    #raw_reviews = raw_dir + 'yelp_academic_dataset_review.json'
    #business_data = raw_dir + 'yelp_academic_dataset_business.json'
    #split_business_by_state(business_data, outfile=parsed_dir + 'businesses')
    #get_restaurants('../data/parsed/businesses_WI')
    #get_reviews_for_businesses(parsed_dir + 'businesses_WI_restaurants', raw_reviews)
    #get_attributes('../data/yelp_data/yelp_academic_dataset_business.json', '../data/parsed/attributes')
    '''
    Creates a bag of words representation based on the WI restaurants and reviews
    '''
    bag_of_words = md.create_bag_of_wods(parsed_dir + 'businesses_WI_restaurants',
                                         parsed_dir + 'businesses_WI_restaurants_reviews', "Price Range")
    bag_of_words.make_sparse_datamtrix()
    
    '''
    Used this for the development of the One Vs Rest model. Want to confer on architecture of program before I move things around
    Figured it'd either be like this or maybe Class setup that calls the method and prints out internally.
    '''
    #print(bag_of_words.labels[0:10])
    #X_train, X_test, y_train, y_test = train_test_split(bag_of_words.datamatrix, bag_of_words.labels, test_size=0.3)
    #one_v_rest = mlm.one_vs_rest(X_train,y_train)
    #first10Predict = one_v_rest.predict(bag_of_words.datamatrix[0:10])
    #print (first10Predict)
    #print(one_v_rest.score(X_test,y_test))
    
    # get_reviews_for_state('../data/parsed/businesses_TX', '../data/yelp_data/yelp_academic_dataset_review.json')
