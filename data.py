import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import torch
from torch.utils.data import Dataset, DataLoader

# import data 
data = pd.read_csv('PoetryFoundationData.csv')
data = data.dropna()

# Delete the first column
data = data.drop(data.columns[0], axis=1)

# in the titles, drop all '\n\n' occurrences
data['Title'] = data['Title'].apply(lambda x: x.replace('\n\n', ''))


# check how many poems contain a ';' in the text
data['Poem'].apply(lambda x: '<LINE>' in x).sum()

# remove this
data['Poem'] = data['Poem'].apply(lambda x: x.replace('<LINE>', ''))

# replace all \n with a '<LINE>' character
data['Poem'] = data['Poem'].apply(lambda x: x.replace('\n', '<LINE>'))
# replace all double <LINE><LINE> with a single <LINE>
data['Poem'] = data['Poem'].apply(lambda x: x.replace('<LINE><LINE>', '<LINE>'))
# remove all leading and trailing <LINE> characters
data['Poem'] = data['Poem'].apply(lambda x: x.strip('<LINE>'))

# set all poems to lowercase
data['Poem'] = data['Poem'].apply(lambda x: x.lower())

# sometimes there are multiple spaces between words, replace them with a single space
data['Poem'] = data['Poem'].apply(lambda x: ' '.join(x.split()))

# set all tags to lowercase
data['Tags'] = data['Tags'].apply(lambda x: x.lower())

# remove all leading and trailing spaces
data['Tags'] = data['Tags'].apply(lambda x: x.strip())


# add poem and tags to new dataframe
poems = pd.DataFrame()
poems['Poem'] = data['Poem']
poems['Tags'] = data['Tags']

# save the poems to a new csv file
poems.to_csv('poems.csv')



class DataProcessor(object):
    def __init__(self, ):
        super().__init__()
        nlp = spacy.load("en_core_web_sm")
        nltk.download('omw-1.4')
        nltk.download("punkt")
        nltk.download("wordnet")
        nltk.download("stopwords")

    @staticmethod
    def preprocess_text(text):
        # Tokenize, remove punctuation and lowercase
        try:
            tokens = nltk.word_tokenize(text)
        except TypeError as e:
            print("Error in tokenizing text \"%s\": %s", text, str(e))
            return ""

        tokens = [word.lower() for word in tokens if word.isalpha()]

        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        processed_text = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]

        return " ".join(processed_text)

    def process_batch(self, texts):
        return [self.preprocess_text(d) for d in texts]
import re
import pandas as pd

def find_special_characters(text):
    # Regular expression to find special characters excluding letters, digits, whitespace, punctuation, and <, >
    special_characters = re.findall(r'[^a-zA-Z0-9\s.,!?;:()\'\"-<>]', text)
    return special_characters

# Load your data
data = pd.read_csv('poems.csv')
data.dropna(inplace=True)

# Run function to find special characters for all instances in the data
special_lists = data['Poem'].apply(lambda x: find_special_characters(x))
data.dropna(inplace=True)

# Combine all lists into one list and ensure all values are unique
all_special_characters = set([char for sublist in special_lists for char in sublist])

# Convert the set back to a list if needed
unique_special_characters = list(all_special_characters)

class Tokenizer(object):
    def __init__(self, max_length=0, special_characters=[]):
        super().__init__()

        self.max_length = max_length
        self.special_characters = special_characters
        self.alphabet_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        self.alphabet = self.prepare_alphabet()
        self.decoded_alphabet = self.prepare_decoded_alphabet()

    def prepare_alphabet(self):
        # PREPARE THE ALPHABET (CHAR->INT)
        # as a dictionary
        alphabet = {}
        alphabet['pad'] = 0  # add 'pad'
        count = 1

        for letter in self.alphabet_letters:
            alphabet[letter] = count
            count += 1

        # add ' ', 'cls' tokens
        alphabet[' '] = count
        alphabet['cls'] = count + 1

        return alphabet

    def prepare_decoded_alphabet(self):
        # PREPARE DECODED ALPHABET (INT->CHAR)
        decoded_alphabet_ints = [i for i in range(len(self.alphabet_letters))]

        decoded_alphabet = {}
        decoded_alphabet[0] = 'pad'

        for i in decoded_alphabet_ints:
            decoded_alphabet[i+1] = self.alphabet_letters[i]

            decoded_alphabet[i+2] = ' '
        decoded_alphabet[i+3] = 'cls'

        return decoded_alphabet

    def encode(self, texts):
        N = len(texts)

        if self.max_length == 0:
            max_length = 0
            for i in range(N):
                len_i = len(texts[i])
                if len_i > max_length:
                    max_length = len_i
        else:
            max_length = self.max_length

        tokens = np.zeros((N, max_length+1))

        for i in range(N):
            len_i = len(texts[i])
            for j in range(-1, max_length):
                if j == -1:
                    tokens[i,j+1] = self.alphabet['cls']
                elif j >= len_i:
                    tokens[i,j+1] = self.alphabet['pad']
                else:
                    if texts[i][j] == 'é':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'í':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'á':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ó':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'æ':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ä':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ū':
                        tokens[i,j+1] = self.alphabet['u']
                    elif texts[i][j] == 'à':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ç':
                        tokens[i,j+1] = self.alphabet['c']
                    elif texts[i][j] == 'ë':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'ñ':
                        tokens[i,j+1] = self.alphabet['n']
                    elif texts[i][j] == 'ö':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'ü':
                        tokens[i,j+1] = self.alphabet['u']
                    elif texts[i][j] == 'ú':
                        tokens[i,j+1] = self.alphabet['u']
                    elif texts[i][j] == 'û':
                        tokens[i,j+1] = self.alphabet['u']
                    elif texts[i][j] == 'å':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'œ':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'ß':
                        tokens[i,j+1] = self.alphabet['s']
                    elif texts[i][j] == 'å':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ø':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'è':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'ï':
                        tokens[i,j+1] = self.alphabet['i']
                    elif texts[i][j] == 'â':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ê':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'î':
                        tokens[i,j+1] = self.alphabet['i']
                    elif texts[i][j] == 'ô':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'ō':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'ā':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ī':
                        tokens[i,j+1] = self.alphabet['i']
                    elif texts[i][j] == 'ē':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'ồ':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'ế':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'π':
                        tokens[i,j+1] = self.alphabet['p']
                    elif texts[i][j] == '∞':
                        tokens[i,j+1] = self.alphabet['i']
                    elif texts[i][j] == '∑':
                        tokens[i,j+1] = self.alphabet['s']
                    elif texts[i][j] == '√':
                        tokens[i,j+1] = self.alphabet['r']
                    elif texts[i][j] == '∫':
                        tokens[i,j+1] = self.alphabet['i']
                    elif texts[i][j] == '≈':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'ﬂ':
                        tokens[i,j+1] = self.alphabet['f']
                    elif texts[i][j] == 'ﬁ':
                        tokens[i,j+1] = self.alphabet['f']
                    elif texts[i][j] == 'ﬀ':
                        tokens[i,j+1] = self.alphabet['f']
                    elif texts[i][j] == 'ﬃ':
                        tokens[i,j+1] = self.alphabet['f']
                    elif texts[i][j] == 'α':
                        tokens[i,j+1] = self.alphabet['a']
                    elif texts[i][j] == 'β':
                        tokens[i,j+1] = self.alphabet['b']
                    elif texts[i][j] == 'γ':
                        tokens[i,j+1] = self.alphabet['g']
                    elif texts[i][j] == 'δ':
                        tokens[i,j+1] = self.alphabet['d']
                    elif texts[i][j] == 'ε':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'ζ':
                        tokens[i,j+1] = self.alphabet['z']
                    elif texts[i][j] == 'η':
                        tokens[i,j+1] = self.alphabet['e']
                    elif texts[i][j] == 'θ':
                        tokens[i,j+1] = self.alphabet['t']
                    elif texts[i][j] == 'ι':
                        tokens[i,j+1] = self.alphabet['i']
                    elif texts[i][j] == 'κ':
                        tokens[i,j+1] = self.alphabet['k']
                    elif texts[i][j] == 'λ':
                        tokens[i,j+1] = self.alphabet['l']
                    elif texts[i][j] == 'μ':
                        tokens[i,j+1] = self.alphabet['m']
                    elif texts[i][j] == 'ν':
                        tokens[i,j+1] = self.alphabet['n']
                    elif texts[i][j] == 'ξ':
                        tokens[i,j+1] = self.alphabet['x']
                    elif texts[i][j] == 'ο':
                        tokens[i,j+1] = self.alphabet['o']
                    elif texts[i][j] == 'π':
                        tokens[i,j+1] = self.alphabet['p']
                    elif texts[i][j] == 'ρ':
                        tokens[i,j+1] = self.alphabet['r']
                    elif texts[i][j] == 'σ':
                        tokens[i,j+1] = self.alphabet['s']
                    elif texts[i][j] == 'τ':
                        tokens[i,j+1] = self.alphabet['t']
                    elif texts[i][j] == 'υ':
                        tokens[i,j+1] = self.alphabet['u']
                    elif texts[i][j] == 'φ':
                        tokens[i,j+1] = self.alphabet['f']
                    elif texts[i][j] == 'χ':
                        tokens[i,j+1] = self.alphabet['c']
                    elif texts[i][j] == 'ψ':
                        tokens[i,j+1] = self.alphabet['p']
                    elif texts[i][j] == 'ω':
                        tokens[i,j+1] = self.alphabet['w']
                    elif texts[i][j] in self.special_characters:
                        tokens[i,j+1] = self.alphabet['q']
                    else:
                        tokens[i,j+1] = self.alphabet[texts[i][j]]

        return tokens

    def decode(self, tokens):
        texts = []

        for i in range(len(tokens)):
            tokens_i = tokens[i,:]
            text_i = ''
            for j in range(len(tokens_i)):
                if tokens_i[j] == 0:
                    break
                else:
                    if self.decoded_alphabet[tokens_i[j]] != 'cls':
                        text_i += self.decoded_alphabet[tokens_i[j]]
            texts.append(text_i)

        return texts
dataprocessor = DataProcessor()
tokenizer = Tokenizer(max_length=1200, special_characters=unique_special_characters)

# randomly split the data into training, test and validation sets
data = pd.read_csv('poems.csv')
data.dropna(inplace=True)

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# split the data into training, test and validation sets
train_data = data[:int(0.7*len(data))]
test_data = data[int(0.7*len(data)):int(0.85*len(data))]
val_data = data[int(0.85*len(data)):]
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')
val_data.to_csv('val_data.csv')

# process the data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')
val_data = pd.read_csv('val_data.csv')

train_poems = dataprocessor.process_batch(train_data['Poem'])
test_poems = dataprocessor.process_batch(test_data['Poem'])
val_poems = dataprocessor.process_batch(val_data['Poem'])

train_tokens = torch.from_numpy(tokenizer.encode(train_poems)).long()
test_tokens = torch.from_numpy(tokenizer.encode(test_poems)).long()
val_tokens = torch.from_numpy(tokenizer.encode(val_poems)).long()
