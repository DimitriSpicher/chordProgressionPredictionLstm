import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import random

class LstmChordPredictor(nn.Module):
    def __init__(self, n_hidden=100):
        super(LstmChordPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)
    
    def forward(self,sequence):
        
        outputs = []
        n_samples = 1

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        
        for input_t in sequence.split(1, dim=1):

            h_t, c_t = self.lstm1(input_t,(h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t,(h_t2, c_t2))
            output = self.linear(h_t2)

            outputs.append(output)

        return torch.cat(outputs, dim=1)

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
    token = np.arange(0, len(chordSet))
    vocabulary = dict(zip(chordSet,token))

    return vocabulary

def splitData(chordSequences):
    random.shuffle(chordSequences)
    splitindex = int((len(chordSequences)*0.8))
    trainSequences = chordSequences[:splitindex]
    testSequences = chordSequences[splitindex:]

    return trainSequences, testSequences


def training_loop(trainDataLoader, model, optimizer, criterion, epochs):

    trainLosses = []

    for epoch in epochs:
        sequence = next(iter(trainDataLoader))
        
        optimizer.zero_grad()
        output = model(sequence)

        trainLoss = criterion(output.squeeze()[:-1], sequence.squeeze()[1:])
        print(f"epoch: {epoch}")
        print(f"trainLoss: {trainLoss}")
        trainLosses.append(trainLoss)

        trainLoss.backward()

        optimizer.step()

    return trainLosses


def test_loop(testDataLoader, model, criterion):

    testLosses = []

    for testSequence in testDataLoader:
        output = model(testSequence)
        testLoss = criterion(output.squeeze()[:-1], testSequence.squeeze()[1:])
        print(f"TestLoss: {testLoss}")
        testLosses.append(testLoss)

    return testLosses

def main(path):
    learningRate = 0.003
    chordSequences = load_data(path)

    vocabulary = buildVocabulary(chordSequences)

    trainSequences, testSequences = splitData(chordSequences)

    trainData = chordDataset(chordSequences=trainSequences, vocabulary=vocabulary)
    testData = chordDataset(chordSequences=testSequences, vocabulary=vocabulary)
    trainDataLoader = DataLoader(trainData, batch_size=1)
    testDataLoader = DataLoader(testData, batch_size=1)

    model = LstmChordPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    epochs = np.arange(0, len(trainSequences))
    
    trainLoss = training_loop(trainDataLoader, model, optimizer, criterion, epochs)
    testLoss = test_loop(testDataLoader, model, criterion)

    return


if __name__=='__main__':
    path = r"D:\aMUZE\DeepLearning\LstmChordProgressions\LstmData\chord_sentences.txt"
    main(path)
