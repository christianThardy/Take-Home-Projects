# I need random state on all stochastic functions

# Stdlib
import json
import random
import math
from math import sqrt

from itertools import groupby
from itertools import chain


# Third party
import numpy as np

import spacy
from spacy.gold import GoldParse
from spacy import displacy

from nltk.tokenize import sent_tokenize

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt



"""
Importance of stopwords comes down to downstream goals/needs

Depending on the type of analysis, maybe capitalization signifys importance or non-importance in resumes
"""

stopwords=['ourselves', 'hers','the','of','and','in', 'between', 'yourself', 'but', 'again','of',
           'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 
           'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 
           'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 
           'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 
           'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 
           'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 
           'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 
           'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 
           'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 
           'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'city', 'state', 
	   'company', 'experience', 'months', 'Company', 'City', 'State', 'Name', '1', 'A', 'project', 
	   'team', 'customer', 'I', 'new', '6', 'year', 'using', 'work', 'also', 'less', 'getting', 
	   'involved', 'also', 'Service', 'including', 'various', 'C', '3', '2']



# Loads word count corpus
def read_1w_corpus(name, sep="\t"):
    for line in open(name, encoding = 'utf-8'):
        yield line.split(sep)



"""
Creates two new columns in the d dataframe: 'colors' and 'sizes'. Colors takes 
a dataframe row as input, returns a color based on the values in the 
skill_sentence_words_developer and skill_sentence_words_hr columns
 
If skill_sentence_words_developer == True, returns the color red; if 
skill_sentence_words_hr is True, returns the color blue; otherwise, it returns the color green

Sizes function also takes a dataframe row and returns a size based on the values 
in the skill_sentence_words_developer and skill_sentence_words_hr columns. If 
either column is True, returns 100, otherwise, returns 1. The apply method applies 
the colors and sizes functions to each row in the d dataframe, resulting values are 
assigned to the 'colors' and 'sizes' columns 
"""

def colors(row):
    if row['skill_sentence_words_developer']:
        return '#FF0000'
    if row['skill_sentence_words_hr']:
        return '#0000FF'
    else:
        return '#8FBC8F'


def sizes(row):
    if row['skill_sentence_words_developer'] or row['skill_sentence_words_hr']:
        return 100
    else:
        return 1


# Average/max character length comparison for resume categories dataset and annotated dataset
def analyze_characters(df, column_name):
    df['ncharacters'] = df[column_name].str.len()
    avg_char = round(df['ncharacters'].mean())
    max_char = round(df['ncharacters'].max())
    print('Average length: {}'.format(avg_char))
    print('Max length: {}'.format(max_char))


# Average/max word count comparison for resume categories dataset and annotated dataset
def analyze_words(df, column_name):
    df['nwords'] = df[column_name].apply(lambda x: len(x.split()))
    avg_words = round(df['nwords'].mean())
    max_words = round(df['nwords'].max())
    print('Average count{}'.format(avg_words))
    print('Max count: {}'.format(max_words))


# Average/max sentence count and length comparison for resume categories dataset and annotated dataset
def analyze_sentence_count_len(df, column_name):
    df['sent_csv_count'] = df[column_name].apply(lambda x: len(sent_tokenize(x)))
    avg_sent_csv = round(df['sent_csv_count'].mean())
    print('Average Sentence Count: {}'.format(avg_sent_csv))

    df['avg_sent_csv_len'] = df[column_name].apply(lambda x: np.mean([len(w.split()) for w in sent_tokenize(x)]))
    avg_sent_csv_len = round(np.mean(df['avg_sent_csv_len']))
    print('Average Sentence Length: {}'.format(avg_sent_csv_len))



# Takes a dataframe and a list of stopwords as input and returns a list of all 
# the words in the dataframe that are not in the stopwords list

def get_words(df, stopwords):
    allwords = []
    for word in df:
        lo_w = []
        list_of_words = str(word).split()
        for w in list_of_words:
            if w not in stopwords:
                lo_w.append(w)
        allwords += lo_w
    return [word for word in allwords if word != "'s"]



"""
Takes in a string representing a resume and processes it to remove 
certain elements and formatting removes URLs, RT and cc, hashtags, and mentions from the resume 
using regex. Also removes punctuation marks and non-ASCII characters. Replaces multiple 
consecutive whitespaces with a single space
"""

def clean_resume(category_resume):
    category_resume = re.sub('http\S+\s*', ' ', category_resume)  # remove URLs
    category_resume = re.sub('RT|cc', ' ', category_resume)  # remove RT and cc
    category_resume = re.sub('#\S+', '', category_resume)  # remove hashtags
    category_resume = re.sub('@\S+', '  ', category_resume)  # remove mentions
    category_resume = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', category_resume)  # remove punctuations
    category_resume = re.sub(r'[^\x00-\x7f]',r' ', category_resume) 
    category_resume = re.sub('\s+', ' ', category_resume)  # remove extra whitespace
    return category_resume


"""
Defines a function that takes in two PileDistribution objects, piledist1 and 
piledist2, and a boolean normalize flagReturns three lists: positions, p1_signature, 
and p2_signature

The positions list is created by combining the positions of all piles in piledist1 
and piledist2, removing duplicates, and sorting the positions by their distance from the origin (0,0)

The p1_signature and p2_signature lists are created by iterating through the positions 
list. For each position, the function checks if that position has a mass in piledist1 
and piledist2. If it does, the mass at that position is added to the corresponding 
signature list. If it does not, a mass of 0 is added to the list. If normalize is True, 
masses in the signature lists are divided by the sum of all masses in the list
"""

def generate_signatures(piledist1, piledist2, normalize=False):
    
    # build unique list of pile positions
    # sorted by distance from the origin
    all_piles = piledist1.piles + piledist2.piles
    positions = sorted(list(set(pile.position for pile in all_piles)),
                       key=lambda x: sqrt(x[0]**2 + x[1]**2))

    # Build signatures
    # Check if the distribution has a mass at this position or return 0
    p1_signature = []
    p2_signature = []
    for position in positions:
        p1_location_mass = piledist1.masses.get(position, 0)
        p2_location_mass = piledist2.masses.get(position, 0)
        p1_signature.append(p1_location_mass)
        p2_signature.append(p2_location_mass)
    if normalize:
        p1_signature = [mass / sum(p1_signature) for mass in p1_signature]
        p2_signature = [mass / sum(p2_signature) for mass in p2_signature]
    
    return positions, p1_signature, p2_signature



"""
Plots a solution to an Earth Movers Distance (EMD) problem. The function takes in six arguments:

pd1: the first pile distribution object
pd2: the second pile distribution object
positions: a list of positions where piles exist in both distributions
emd: the EMD value between the two distributions
flow: the flow matrix between the two distributions
normed: a boolean value indicating whether the masses in the distributions are normalized
r_scale: a scaling factor for the size of the plotted piles
figsize: a tuple indicating the size of the plot
annotate: a boolean value indicating whether to annotate the plot with the EMD value

Defines variables p1_x, p1_y, p2_x, and p2_y as the x-coordinates and 
y-coordinates of the piles in the first and second distributions. Determines whether 
the masses of the piles are normalized and sets the variables p1_masses and p2_masses

Then iterates over the nonzero elements in the flow matrix, creating a 
dictionary measure with keys 'to', 'from', 'xs', 'ys', and 'value', which represent 
the positions of the piles involved in the flow, the x-coordinates and y-coordinates 
of the positions, and the value of the flow. The function appends measure to the list 
flow_measures

Finally, the function plots the piles in the two distributions as scatter plots,
with the size of each pile proportional to its mass, and plots a line between the 
two piles for each flow in flow_measures. The function also annotates the plot with 
the EMD value if annotate is True
"""

def plot_emd_solution(pd1, pd2, positions, emd, flow, normed=True, r_scale=5000, figsize=(8,8), annotate=True):
    p1_x = [pile.x for pile in pd1.piles]
    p1_y = [pile.y for pile in pd1.piles]

    p2_x = [pile.x for pile in pd2.piles]
    p2_y = [pile.y for pile in pd2.piles]
    
    if normed:
        p1_masses = [pile.mass / pd1.mass_sum for pile in pd1.piles]
        p2_masses = [pile.mass / pd2.mass_sum for pile in pd2.piles]
    else:
        p2_masses = [pile.mass for pile in pd2.piles]
        p1_masses = [pile.mass for pile in pd1.piles]
        
    flow_measures = []
    for to_pos_ix, from_pos_ix in zip(*np.nonzero(flow)):
        to_pos = positions[to_pos_ix]
        from_pos = positions[from_pos_ix]
        measure = {'to' : to_pos, 'from' : from_pos,
                   'xs' : (to_pos[0], from_pos[0]),
                   'ys' : (to_pos[1], from_pos[1]),
                   'value' : flow[to_pos_ix, from_pos_ix]}
        flow_measures.append(measure)

    plt.figure(figsize=figsize)

    plt.scatter(x=p1_x, y=p1_y, s=[r_scale*m for m in p1_masses], c='r', alpha=0.8)   
    plt.scatter(x=p2_x, y=p2_y, s=[r_scale*m for m in p2_masses], c='b', alpha=0.8)

    for measure in flow_measures:
        plt.plot([*measure['xs']], [*measure['ys']],
                 color='black', lw=measure['value']*r_scale/100, alpha=0.7, solid_capstyle='round')



"""
Converts a JSON file containing annotated data into a format that can be used by 
spaCy to train a model. The file path of a JSON file is taken as input. The function then cleans 
the extracted entities by removing leading and trailing white spaces, and appends the text and 
entities to a list called training_data

Trim_entity_spans iterates over each text-entity pair and removes any leading or trailing white 
spaces from the entity's start and end indices
"""

def convert_rh_annotated_data_to_spacy(rh_annotated_data_JSON_FilePath):
    training_data = []
    lines=[]
    with open(rh_annotated_data_JSON_FilePath, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        data = json.loads(line)
        #text = data['content'].replace("\n", " ")
        #text = data['content'].replace("•", "")
        text = data['content']
        entities = []
        data_annotations = data['annotation']
        if data_annotations is not None:
            for annotation in data_annotations:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']

                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    point_start = point['start']
                    point_end = point['end']
                    point_text = point['text']

                    lstrip_diff = len(point_text) - len(point_text.lstrip())
                    rstrip_diff = len(point_text) - len(point_text.rstrip())
                    if lstrip_diff != 0:
                        point_start = point_start + lstrip_diff
                    if rstrip_diff != 0:
                        point_end = point_end - rstrip_diff
                    entities.append((point_start, point_end + 1 , label))
        training_data.append((text, {"entities" : entities}))
    return training_data


def trim_entity_spans(data: list) -> list:
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data


def train_test_split(data, test_size, random_state):

    random.Random(random_state).shuffle(data)
    test_idx = len(data) - math.floor(test_size * len(data))
    train_set = data[0: test_idx]
    test_set = data[test_idx: ]

    return train_set, test_set


"""
Creates a blank spaCy model and trains a spaCy model for named entity recognition (NER)

The function first creates a blank spaCy model nlp and adds a NER pipeline 
to it. Then it adds the labels that the model should recognize to the NER 
pipeline by iterating over the training data and adding each label to the pipeline.

Training is done through iterating over the annotated data, shuffling it randomly, 
and using the nlp.update method to update the model's weights based on the text and 
annotations in the training data. The drop parameter is used to specify the dropout rate, 
which prevents the model from overfitting. The sgd parameter specifies the optimizer 
to use for updating the weights, and the losses parameter records the loss after each update
"""

def train_spacy():
    
    nlp = spacy.blank('en') 
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
        
    for _, annotations in train_data:
         for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  
        optimizer = nlp.begin_training()
        for itn in range(15): 
            print("Starting epoch " + str(itn))
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                nlp.update(
                    [text],  
                    [annotations],
                    drop=0.186005392754230048,  
                    sgd=optimizer,  
                    losses=losses)
            print(losses)
    return nlp



"""
Takes in two arguments: nlp and text. The nlp argument is a pre-trained spaCy model, 
and text is a string of text that the model will be used to analyze

Iterates over the list of tokens, grouping them by their entity label using the 
groupby function. For each group of tokens, the function extracts the start and 
end indices of the entity span, using the index of the first token in the group 
as the start index and the index of the last token plus the length of the token 
text as the end index. The function then appends a tuple containing the start index, 
end index, and entity label to a list of entities
"""

def doc_to_bilou(nlp, text):
    
    doc = nlp(text)
    tokens = [(tok.text, tok.idx, tok.ent_type_) for tok in doc]
    entities = []
    for entity, group in groupby(tokens, key=lambda t: t[-1]):
        if not entity:
            continue
        group = list(group)
        _, start, _ = group[0]
        word, last, _ = group[-1]
        end = last + len(word)
        
        entities.append((
                start,
                end,
                entity
            ))

    gold = GoldParse(nlp(text), entities = entities)
    pred_ents = gold.ner
    
    return pred_ents



"""
Evaluates the performance of the NER model
Takes two lists of BIO-encoded sequences as input, y_true and y_pred, which represent 
the true and predicted labels for a set of text sequences

LabelBinarizer transforms the lists of BIO-encoded sequences into binary matrices,
then calculates evaluation metrics precision, recall, and f1-score for each class in the dataset
Also calculates the overall accuracy of the model
"""

def ner_report(y_true, y_pred):
    """
    Classification report computes token-level metrics and discards "O" labels.
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset
    ), accuracy_score(y_true_combined, y_pred_combined)
    



"""
The DirtPile class represents a dirt pile on a two-dimensional plane

Attributes are the position, mass, and label of the dirt pile

The position and mass arguments are required and are used to initialize the position, 
x, y, and mass attributes. The label argument is optional and is used to initialize the 
label attribute.

https://en.wikipedia.org/wiki/Earth_mover%27s_distance
"""

class DirtPile():
    def __init__(self, position, mass, label=None):
        self.position = position
        self.x, self.y = position
        self.mass = mass
        if not label:
            self.label = "{mass} @ ({x}, {y})".format(mass=self.mass, x=self.x, y=self.y)
        else:
            self.label = label
            
    def __str__(self):
        return "{mass} @ ({x}, {y})".format(mass=self.mass, x=self.x, y=self.y)



"""
The PileDistribution class represents a distribution of dirt piles on a 
two-dimensional plane, with attributes for the piles, their masses, and 
the sum of their masses, and methods for representing the distribution as 
a string and accessing the dirt piles in the distribution.
"""

class PileDistribution():
    def __init__(self, *piles):
        self.piles = list(piles)
        self.masses = {tuple(p.position): p.mass for p in self.piles}
        self.mass_sum = sum(p.mass for p in self.piles)
        
    def __str__(self):
        return '\n'.join([str(pile) for pile in self.piles])
    
    def __getitem__(self, index):
        return self.piles[index]