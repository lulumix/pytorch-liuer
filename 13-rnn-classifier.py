


'''
if __name__ == '__main__':
    classfier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device('cuda')
        classfier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classfier.parameters(), lr=LR)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(N_EPOCHS):
        trainModel()
        acc = testModel()
        acc_list.append(acc)
'''

