import os
import json
from business import Business
import modeldata as md
import random as rd
import copy as cp

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
                print ('adding new state {}'.format(state))
                o = outfile + '_' + state
                outfiles[state] = open(o, 'w')
            outfiles[state].write(line)
    for v in iter(outfiles.values()):
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
            for k, v in iter(attributes.items()):
                if k not in attr_hash:
                    attr_hash[k] = []
                values = []
                if not isinstance(v, dict):
                    values.append(v)
                else:
                    for val in iter(v.keys()):
                        values.append(val)
                for value in values:
                    if value not in attr_hash[k]:
                        attr_hash[k].append(value)
    with open(outfile, 'w') as o:
        for k, v in iter(attr_hash.items()):
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
'''
number_set: 1-?
train_weight: 1-9. 
'''
def split_data(businessfile, reviewfile, number_sets, train_weight):
    all_sets = {}
    for i  in range(1, number_sets+1):
        all_sets["train" + str(i)] = {}
        all_sets["test" + str(i)] = {}
        
    with open(businessfile) as rst:
        for line in rst:
            jline = json.loads(line)
            b_id = jline['business_id']
            set_num = rd.randint(1, number_sets)
            isTrain = rd.randint(1, 10)
            if isTrain > train_weight:
                dict_type = "test"
            else:
                dict_type = "train"
            dict_temp = {}    
            if (dict_type + str(set_num)) in all_sets.keys():
                dict_temp = all_sets[dict_type + str(set_num)] 
            dict_temp[b_id] = Business(jline)
            all_sets[dict_type + str(set_num)]  = dict_temp
            
    all_keys = all_sets.keys()            
    with open(reviewfile) as rvw:
        for line in rvw:
            jline = json.loads(line)
            b_id = jline['business_id']
            for i_dict in all_keys:
                if b_id in all_sets[i_dict].keys():
                    dict_temp = all_sets[i_dict]
                    dict_temp[b_id].add_review(jline)
                    all_sets[i_dict]  = dict_temp 
    return all_sets



def filter(data, word_lmt = 0, review_lmt = 0):
    bid = data.keys()
    dataOut = cp.deepcopy(data)
    count_wd = 0
    print(len(dataOut))
    for id in bid:
        for review in data[id].reviews:
            count = len(review['text'].split(' '))
            count_wd = count_wd + count

        if len(data[id].reviews) < review_lmt or count_wd < word_lmt:
            del dataOut[id]
            print(len(dataOut))
    count_wd = 0
    return dataOut


if __name__ == '__main__':
    data_dir = '../data/'
    parsed_dir = data_dir + 'parsed/'
    raw_dir = data_dir + 'yelp_data/'
    '''
    some setting up
    '''
    ''' 
    raw_reviews = raw_dir + 'yelp_academic_dataset_review.json'
    business_data = raw_dir + 'yelp_academic_dataset_business.json'
    split_business_by_state(business_data, outfile=parsed_dir + 'businesses')
    get_restaurants('../data/parsed/businesses_WI')
    get_reviews_for_businesses(parsed_dir + 'businesses_WI_restaurants', raw_reviews)
    get_attributes('../data/yelp_data/yelp_academic_dataset_business.json', '../data/parsed/attributes')
    '''
    '''
    Creates a bag of words representation based on the WI restaurants and reviews
    '''
    all_sets = split_data(parsed_dir + 'businesses_WI_restaurants.json', parsed_dir + 'businesses_WI_restaurants_reviews.json', 2, 5)
    filter(all_sets['test1'], 1000, 5)
    #bag_of_words = md.create_bag_of_wods(all_sets['train1'], "Price Range")
    #bag_of_words.make_sparse_datamtrix()

    #json_data = open(parsed_dir + 'businesses_WI_restaurants').read()
    #data = json.load(json_data)
 # get_reviews_for_state('../data/parsed/businesses_TX', '../data/yelp_data/yelp_academic_dataset_review.json')
