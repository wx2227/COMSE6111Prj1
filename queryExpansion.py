import math
import sys
from functools import partial
import requests
import numpy as np
import nltk
from collections import defaultdict, OrderedDict

nltk.download('wordnet')
from string import punctuation

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#  we use a combination of rocchio with bigram re-scoring
# the documents consist of the search results' title and snippet
class QueryExpansion:

    def __init__(self, cx, key, precision, query):
        self.cx = cx
        self.key = key
        self.precision = precision
        self.queryList = query.split(" ")
        # documents that labeled as 'Y'
        self.relevant = []
        # documents that labeled as N'
        self.irrelevant = []
        # all documents exclude non-html files
        self.allDocs = []
        # all words that appear in the search results' title and snippet
        self.docsSet = set()

        self.lemma = nltk.wordnet.WordNetLemmatizer()
        self.bigramCount = {}

        self.relcount = 0
        self.irrelcount = 0

    # retrieve the top-10 search results from google's api
    # @return list of dict
    def retrieveResult(self):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"cx": self.cx,
                  "q": " ".join(self.queryList),
                  "key": self.key,
                  "num": 10}

        response = requests.get(url=url, params=params).json()

        data = response["items"]
        return data

    # load the stopwords and the punctuations
    # @return set of string
    def loadStopwords(self):
        with open("./proj1-stop.txt", "r+") as f:
            words = f.read()
            stopwords = words.split("\n")
            f.close()
        stopwords = set(stopwords).union(set(punctuation))
        return stopwords

    # we extract information from the title and the snippet of the document
    # first tokenize the documents with nltk's tokenizer, and lowercase it
    # then remove the stopwords and punctuations
    # then lemmatize each words
    # @return list of str
    def documentProcess(self, item):
        stopwords = self.loadStopwords()
        title = word_tokenize(item['title'].lower())
        description = word_tokenize(item['snippet'].lower())

        d = []
        for elem in title + description:
            if not elem in stopwords:
                temp = self.lemma.lemmatize(elem)
                d.append(temp)
        return d

    # update the stored documents
    def allDocsSet(self, allDocs):
        for doc in allDocs:
            self.docsSet.update(doc)

    # demonstrate the search result to user and let the user label whether the result is consistent with
    # what they want or not
    # store the labeled document into self.relevant(documents that labeled as 'Y') and
    # self.irrelevant(documents that labeled as 'N')
    # store all the words appear in the search result's title and snippet into self.docsSet
    def resToUser(self, data):
        self.relcount = 0
        self.irrelcount = 0
        self.allDocs = []
        self.docsSet = set()
        for i, item in enumerate(data):
            mark = input("Result " + str(i + 1) + "\n" + 'TITLE: ' + item['title'] + '\nURL: ' + item['formattedUrl']
                         + '\nSUMMARY: ' + item['snippet'] + '\n RELEVANT (Y/N)? \n')
            # contains fileFormat field in the search result means that the result is not in html form
            if "fileFormat" in item.keys():
                continue
            item = self.documentProcess(item)
            self.allDocs.append(item)
            if str(mark.lower().strip()) == 'y':
                self.relcount += 1
                self.relevant.append(item)
                # add bigrams counts for the relevant documents
                for bigram in nltk.bigrams(item):
                    if bigram in self.bigramCount:
                        self.bigramCount[bigram] += 1
                    else:
                        self.bigramCount[bigram] = 1
            else:
                self.irrelcount += 1
                self.irrelevant.append(item)
                # subtract bigram counts for the irrelevant documents
                for bigram in nltk.bigrams(item):
                    if bigram in self.bigramCount:
                        self.bigramCount[bigram] -= 1
                    else:
                        self.bigramCount[bigram] = -1
        self.allDocsSet(self.allDocs)
        self.docsSet.update(self.queryList)

    # calculate the rocchio score of each words
    # @return dict{str: float}
    def rocchio(self):

        alpha = 1
        beta = 0.75
        gamma = 0.15

        q_tf_idf = tf_idf(self.allDocs, self.queryList)
        # expand the tf-idf of the query to the same shape as tf-idf of the documents
        q_tf_idf = np.array([q_tf_idf[self.queryList.index(el)] if el in self.queryList else 0 for el in self.docsSet])

        relevant_tf_idf = tf_idf(self.relevant, self.docsSet)

        irrelevant_tf_idf = tf_idf(self.irrelevant, self.docsSet)

        score = alpha * q_tf_idf + beta * relevant_tf_idf - gamma * irrelevant_tf_idf

        score = dict(zip(self.docsSet, score))

        return score

    # modifies the current query list with rocchio score and bigram
    def newQuery(self):
        score = self.rocchio()
        bigrams = sorted(self.bigramCount.items(), key=lambda x: x[1], reverse=True)
        bigrams = list(filter(lambda x: x[1] > 1, bigrams))  # filter out bigrams with only one occurrence

        # if some word combinations happens a lot in the relevant document
        # it's more likely that they are related to the query (user intent)
        # we thus want to give those combinations more weight based on their occurrence
        for bigram in bigrams:
            occurrence = bigram[1]
            word1, word2 = bigram[0][0], bigram[0][1]

            score[word1] = score.get(word1, 0) * occurrence
            score[word2] = score.get(word2, 0) * occurrence
            # if some relevant word combinations includes our query terms
            # its more likely that combination is close to user intent
            # we give more weight to those combinations
            if word1 in self.queryList:
                score[word2] = score.get(word2, 0) * 2
            if word2 in self.queryList:
                score[word1] = score.get(word1, 0) * 2

        rocchioTop2 = []  # find top 2 possible extra words from rocchio
        count = 0
        while True:
            word = max(score, key=score.get)
            if not word:
                break
            if not word in self.queryList:
                rocchioTop2.append(word)
                count += 1
            if count > 1:
                break
            del score[word]

        appendRemain = 2
        # if the top bigram occurrence is high, we want extract candidate query term from it
        if len(bigrams) >= 1 and bigrams[0][1] >= 5:
            bigram = bigrams[0]
            word1, word2 = bigram[0][0], bigram[0][1]
            if word1 in self.queryList and word2 in self.queryList:  # if both words in list, reorder using bigram
                self.queryList.remove(word1)
                self.queryList.remove(word2)
                self.queryList = [word1, word2] + self.queryList
            elif word1 not in self.queryList and word2 not in self.queryList:  # both word not in, append both
                self.queryList.append(word1)
                self.queryList.append(word2)
                appendRemain -= 2
                return
            else:  # only one word in list, add the other word of the bigram
                if word1 in self.queryList:
                    self.queryList.remove(word1)
                if word2 in self.queryList:
                    self.queryList.remove(word2)
                self.queryList = [word1, word2] + self.queryList
                appendRemain -= 1


        # add candidate term from rocchio
        self.queryList = self.queryList + rocchioTop2[:appendRemain]
        # reorder the whole query using bigram
        for bigram in bigrams[:10]:
            word1, word2 = bigram[0][0], bigram[0][1]
            if word1 in self.queryList and word2 in self.queryList:  # if both words in list, reorder using bigram
                self.queryList.remove(word1)
                self.queryList.remove(word2)
                self.queryList = [word1, word2] + self.queryList




    # for first iteration, if precision is 0,
    # or the total number of documents returned from the search engine is below 10 return nothing
    # @return str
    def firstIteration(self):
        data = self.retrieveResult()
        self.resToUser(data)

        if self.relcount + self.irrelcount < 10 or self.relcount == 0:
            return ""

        return self.queryExpansion()

    # expand the query until the precision is above the target precision
    # print the current query list in each iteration
    def queryExpansion(self):
        while True:
            currPrecision = self.relcount / (self.relcount + self.irrelcount)
            if currPrecision < self.precision:
                # newQuery is to expand and reorder the current query
                self.newQuery()
                print()
                print("Expanded query:", " ".join(self.queryList))
                print()
                # retrieveResult is to get the search result of current query
                data = self.retrieveResult()
                # demonstrate the search result to the user
                self.resToUser(data)

            else:
                break

        return " ".join(self.queryList)

# calculate the term frequency in a certain document
def termFrequency(term, document=[]):
    termCount = document.count(term.lower())
    return math.log(termCount + 1)

# calculate the inverse document frequency in a provided list of documents
def inverseDocumentFrequency(term, docs=[]):
    count = 0
    idf = 0
    for doc in docs:
        if term.lower() in doc:
            count += 1
    if count:
        idf = math.log(len(docs) / count)
    return idf

# compute the term frequency of provided form(list of words)
def tfMatrixCompute(docs, form):
    matrix = []
    for doc in docs:
        tf_t = partial(termFrequency, document=doc)
        tf = list(map(tf_t, form))
        matrix.append(tf)
    matrix = np.array(matrix)
    return matrix

# compute the inverse document frequency of provided form(list of words)
def idfMatrixCompute(docs, form):
    matrix = []
    idf_t = partial(inverseDocumentFrequency, docs=docs)
    idf = list(map(idf_t, form))
    matrix = np.array(idf)
    return matrix

# compute the tf-idf of the provided form(list of words)
def tf_idf(docs, form):
    idf = idfMatrixCompute(docs, form)
    tf = tfMatrixCompute(docs, form)
    tfidf = tf * idf
    tfidf = tfidf / np.sqrt((tfidf * tfidf).sum(axis=1)).reshape((len(tfidf), 1))
    tfidf[np.isnan(tfidf)] = 0
    tfidf = tfidf.sum(axis=0) / len(tfidf)
    return tfidf


if __name__ == "__main__":
    # key = sys.argv[1]
    # cx = sys.argv[2]
    # precision = sys.argv[3]
    # query = sys.argv[4]

    cx = "011931726167723972512:orkup7yeals"
    key = "AIzaSyAg_FedCkdEHFmYwRdkqS5Im2zeOjlrC4Y"
    precision = 0.9
    query = "jaguar"

    qe = QueryExpansion(cx, key, precision, query)

    query = qe.firstIteration()

    print("Final Expanded Query:", query)
