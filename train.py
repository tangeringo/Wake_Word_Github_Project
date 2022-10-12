import torch
from torch import nn
from torch.utils.data import DataLoader

from model import LSTMModel, model_params
from dataset import WakeWordDataset, mel_spectrogram

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input_val, target in data_loader:
        input_val, target = input_val.to(device), target.to(device)

        # calculate loss
        prediction = model(input_val)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # instantiating our dataset object and create data loader

    dataset = WakeWordDataset('DATA/test_data.json', mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    train_data_loader = create_data_loader(dataset, BATCH_SIZE)

    # construct model and assign it to device
    model = LSTMModel(**model_params).to(device)
    print(model)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(model, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    # save model
    # torch.save(model.state_dict(), "feedforwardnet.pth")
    # print("Trained feed forward net saved at feedforwardnet.pth")


# END NOTE:
# The model and the dataset is pretty much done..
# We only need to make sure
#   (1) it is compatible
#   (2) the training process works fine
