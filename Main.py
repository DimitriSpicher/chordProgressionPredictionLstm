import torch
import torch.nn as nn 
import numpy as np
import random

class chordDataset(torch.utils.data.Dataset):
    def __init__(self, chordSequences, vocabulary):
        self.chordSequences = chordSequences
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.chordSequences)

    def __getitem__(self,idx):
        tokenizedSequence = tokenize(self.chordSequences[idx], self.vocabulary)
        return tokenizedSequence
# utils
def load_data(path):
    with open (path) as file:
        fileAsString = file.readlines()[0]
        lines = fileAsString.split("_END_")
        lines.pop()
        chordSequences = []
        for line in lines:
            chordSequence = line.split(" ")
            chordSequence.remove("_START_")
            chordSequence[:] = [chord for chord in chordSequence if chord]
            chordSequences.append(chordSequence)

    return chordSequences

def tokenize(chordSequence, vocabulary):
    tokenizedSequence = torch.Tensor([vocabulary[chord] for chord in chordSequence])
    return tokenizedSequence

def buildVocabulary(chordSequences):

    chordSet = sorted(set([chord for sequence in chordSequences for chord in sequence]))
    token = np.arange(1, len(chordSet)+1)
    vocabulary = dict(zip(chordSet,token))
    vocabulary["<pad>"] = 0

    return vocabulary

def splitData(chordSequences):
    random.shuffle(chordSequences)
    splitindex = int((len(chordSequences)*0.8))
    trainSequences = chordSequences[:splitindex]
    testSequences = chordSequences[splitindex:]

    return trainSequences, testSequences

def main(path):
    chordSequences = load_data(path)

    vocabulary = buildVocabulary(chordSequences)

    trainSequences, testSequences = splitData(chordSequences)

    trainData = chordDataset(chordSequences=trainSequences, vocabulary=vocabulary)
    testData = chordDataset(chordSequences=testSequences, vocabulary=vocabulary)
    
    return


if __name__=='__main__':
    path = r"D:\aMUZE\DeepLearning\LstmChordProgressions\LstmData\chord_sentences.txt"
    main(path)
