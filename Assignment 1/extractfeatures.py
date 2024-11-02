import csv
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

#Implememnting a class to extract Features from a text
class ExtractFeatures:

    def __init__(self, text):
        self.text = text
        #Tokenize sentences and words 
        self.sentences = sent_tokenize(text)
        self.words = word_tokenize(text)

    #Feature 1: Implementing a function to calculate number of sentences
    def calculateNumOfSentences(self) -> int:
        return len(self.sentences)
    
    #Feature 2: Implement a function to  calculate average word length
    def calculateAvgWordLength(self) -> float:
        avgWordLength = sum(len(word) for word in self.words if word.isalpha()) / len(self.words)
        return avgWordLength
    
    #Feature 3: Implement a function to calculate the average sentence length in words
    def calculateAvgSentenceLength(self) ->  float:
        avgSentenceLength = sum(len(word_tokenize(sentence)) for sentence in self.sentences) / self.calculateNumOfSentences()
        return avgSentenceLength
    
    #Feature 4: Implementing a function to calculate number of unique words
    def calculateUniqueWords(self) -> int:
        numOfUniqueWords = len(set(word.lower() for word in self.words if word.isalpha()))
        return numOfUniqueWords
    
    #Feature 5: Implementing a function to calculate the lexical richness
    #Lexical Richness = Ratio of the number of unique words to the  total length of words
    def calculateLexicalRichness(self) -> float:
        lexicalRichness = self.calculateUniqueWords / len(self.words)
        return lexicalRichness
    


    
