import torch 

def buildVocabulary(chordSequences):

    vocabulary = {}
    return vocabulary

def tokenize(chordSequence, vocabulary):
    tokenizedSequence = chordSequence
    return tokenizedSequence

class chordDataset(torch.utils.data.Dataset):
    def __init__(self, chordSequences):
        self.chordSequences = chordSequences
        self.vocabulary = buildVocabulary(self.chordSequences)

    def __len__(self):
        return len(self.chordSequences)

    def __getitem__(self,idx):
        chordSequence = self.chordSequences[idx]
        tokenizedSequence = tokenize(chordSequence, self.vocabulary)
        return chordSequence

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
    print("hello world")
    chordSequences=load_data(path)
    print(chordDataset(chordSequences).__getitem__(2))
    
    return


if __name__=='__main__':
    path = r"D:\aMUZE\DeepLearning\LstmChordProgressions\LstmData\chord_sentences.txt"
    main(path)
