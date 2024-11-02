import csv
import os
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from collections import Counter
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
        lexicalRichness = self.calculateUniqueWords() / len(self.words)
        return lexicalRichness
    
    #Implementing a helper function to gather all extracted features
    def collectExtractedfeatures(self) -> list:
        return [self.calculateNumOfSentences(), self.calculateAvgWordLength(), self.calculateAvgSentenceLength(), self.calculateUniqueWords(), self.calculateLexicalRichness()]
    
    #Implementing a function to print file to CSV
    def printFeaturesToCSV(self, output_file = "TextFeatures.csv"):

        # Get the directory of the current script
        script_directory = os.path.dirname(os.path.abspath(__file__))
    
        # Construct the full path for the output file
        output_path = os.path.join(script_directory, output_file)

        #defining the column headers for CSV
        headers = ["num_sentences", "average_word_length", "average_sentence_length", "unique_words", "lexical_richness"]

        #Extact features for each text
        rows = [self.collectExtractedfeatures()]

        #Print features in CSV file
        with open(output_file, "w", newline = "") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

            print(f"Features extracted and saved to {output_file}")




#Writing a test case for its usage
text = "Throughout the programming labs, we will work on the task of text regression: given a text, predict whether it was written by a human or an AI. The dataset comprises multiple text genres, such as news articles, Wikipedia intro texts, or fanfiction."
extractor = ExtractFeatures(text)
extractor.printFeaturesToCSV("TestFeatures.csv")

    
