import torch 
import numpy as np

def buildVocabulary(chordSequences):

    chordSet = sorted(set([chord for sequence in chordSequences for chord in sequence]))
    token = np.arange(1, len(chordSet)+1)
    vocabulary = dict(zip(chordSet,token))
    vocabulary["<pad>"] = 0

    return vocabulary

def tokenize(chordSequence, vocabulary):
    tokenizedSequence = [vocabulary[chord] for chord in chordSequence]
    return tokenizedSequence

class chordDataset(torch.utils.data.Dataset):
    def __init__(self, chordSequences):
        self.chordSequences = chordSequences
        self.vocabulary = buildVocabulary(self.chordSequences)

    def __len__(self):
        return len(self.chordSequences)

    def __getitem__(self,idx):
        tokenizedSequence = torch.tensor(tokenize(self.chordSequences[idx], self.vocabulary))
        return tokenizedSequence

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


def main(path):
    chordSequences=load_data(path)
    x = chordDataset(chordSequences=chordSequences)
    print((x.__getitem__(3)))
    
    return


if __name__=='__main__':
    path = r"D:\aMUZE\DeepLearning\LstmChordProgressions\LstmData\chord_sentences.txt"
    main(path)
