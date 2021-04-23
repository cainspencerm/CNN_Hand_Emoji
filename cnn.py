"""
Convolutional neural networks take a portion of the image to identify features
that make up an output. When recognizing the numbers in the MNIST dataset,
CNNs follow the philosophy 3Blue1Brown intended to map on the network.
"""

from curtsies.fmtfuncs import red, green, blue, yellow
from cv2 import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import time


OUTPUTS = ['backhand_index_pointing_down',
           'backhand_index_pointing_right',
           'backhand_index_pointing_up',
           'call_me_hand',
           'crossed_fingers',
           'flexed_bicep',
           'hand_with_fingers_splayed',
           'index_pointing_up',
           'left_facing_fist',
           'love-you_gesture',
           'middle_finger',
           'ok_hand',
           'oncoming_fist',
           'pinched_fingers'
           'pinching_hand',
           'raised_back_of_hand',
           'raised_fist',
           'raised_hand',
           'right_facing_fist',
           'sign_of_the_horns',
           'thumbs_down',
           'thumbs_up',
           'victory_hand',
           'vulcan_salute',
           'waving_hand']

# OUTPUTS = ['backhand_index_pointing_down',
#            'backhand_index_pointing_right',
#            'backhand_index_pointing_up',
#            'call_me_hand',
#            'crossed_fingers',
#            'flexed_bicep',
#            'hand_with_fingers_splayed',
#            'index_pointing_up',
#            'left_facing_fist',
#            'love-you_gesture']

DEVICE = 'cuda:0'

MODEL_NAME = f"model-{int(time.time() % 100000)}"
IMAGE_SIZE_X = 200
IMAGE_SIZE_Y = 200

if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(blue("Running on GPU"))
else:
    device = torch.device(DEVICE)
    print(yellow("Running on CPU"))


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(IMAGE_SIZE_X, IMAGE_SIZE_Y).view(-1, 1, IMAGE_SIZE_X, IMAGE_SIZE_Y)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 25)

    def convs(self, x):
        x = functional.max_pool2d(functional.relu(self.conv1(x)), (2, 2))
        x = functional.max_pool2d(functional.relu(self.conv2(x)), (2, 2))
        x = functional.max_pool2d(functional.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)
        return functional.softmax(x, dim=1)


def fwd_pass(net, loss_function, optimizer, x, y, train=False):
    if train:
        net.zero_grad()
        optimizer.zero_grad()

    outputs = net(x)
    matches = [torch.argmax(i).item() == j for i, j in zip(outputs, y)]

    accuracy = matches.count(True) / len(y)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return accuracy, loss


def test(net, loss_function, optimizer, test_x, test_y, size=50):
    rand_start = int(np.random.random() * (len(test_x) - size))
    test_x = test_x[rand_start:rand_start + size]
    test_y = test_y[rand_start:rand_start + size]
    batch_size = 50
    val_acc_sum = 0
    val_loss_sum = 0
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            batch_x = test_x[i:i + batch_size].view(-1, 1, IMAGE_SIZE_X, IMAGE_SIZE_Y).to(device)
            batch_y = test_y[i:i + batch_size].to(device)

            val_acc, val_loss = fwd_pass(net, loss_function, optimizer, batch_x, batch_y)
            val_acc_sum += val_acc
            val_loss_sum += val_loss.item()

    return val_acc_sum / (len(test_x) / batch_size), val_loss_sum / (len(test_x) / batch_size)


def train(net, loss_function, optimizer, train_x, train_y, test_x, test_y):
    batch_size = 100
    epochs = 10
    for epoch in range(epochs):
        count = 0
        val_count = 0
        sum_acc = 0
        sum_loss = 0
        sum_val_acc = 0
        sum_val_loss = 0
        for i in tqdm(range(0, len(train_x), batch_size)):
            batch_x = train_x[i:i + batch_size].view(-1, 1, IMAGE_SIZE_X, IMAGE_SIZE_Y).to(device)
            batch_y = train_y[i:i + batch_size].to(device)
            with torch.enable_grad():
                acc, loss = fwd_pass(net, loss_function, optimizer, batch_x, batch_y, train=True)
                sum_acc += acc
                sum_loss += loss.item()
                count += 1

            if i % 200 == 0:
                val_acc, val_loss = test(net, loss_function, optimizer, test_x, test_y)
                sum_val_acc += val_acc
                sum_val_loss += val_loss
                val_count += 1

        print(yellow(f'EPOCH: {epoch}\t'
                     f'avg acc: {round(sum_acc / count, 5)}\t'
                     f'avg loss: {round(sum_loss / count, 5)}\t'
                     f'avg val acc: {round(sum_val_acc / val_count, 5)}\t'
                     f'avg val loss: {round(sum_val_loss / val_count, 5)}'))


def loadTrainingData():
    training_data = np.load("hands_25.npy", allow_pickle=True)
    np.random.shuffle(training_data)

    x = torch.Tensor([i[0] for i in training_data]).view(-1, IMAGE_SIZE_X, IMAGE_SIZE_Y)
    x /= 255.  # Change pixel value to range[0, 1)
    y = [np.argwhere(i[1] == 1.)[0][0] for i in training_data]
    y = torch.LongTensor(y)

    VAL_PCT = 0.1
    val_size = int(len(x) * VAL_PCT)

    train_x = x[:-val_size]
    train_y = y[:-val_size]

    test_x = x[-val_size:]
    test_y = y[-val_size:]

    return train_x, train_y, test_x, test_y


def test_webcam(net, mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, image = cam.read(cv2.IMREAD_GRAYSCALE)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = img[:, img.shape[0] // 2: (img.shape[0] // 2) + img.shape[1]]
        img = cv2.resize(img, (IMAGE_SIZE_X, IMAGE_SIZE_Y))
        cv2.imshow('ai', img)
        img = img / 255.
        # img = np.array(img, dtype=np.double)
        img = torch.as_tensor(img, dtype=torch.float)
        img = img.view(-1, 1, IMAGE_SIZE_X, IMAGE_SIZE_Y)

        with torch.no_grad():
            output = net(img)
            if torch.max(output) > 0.95:
                out = torch.argmax(output)
                for output in OUTPUTS:
                    print(red(output))
                if out < len(OUTPUTS):
                    print(OUTPUTS[int(out)])

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def test_image(net, name):
    _name = name
    while True:
        try:
            image = cv2.imread(_name, cv2.IMREAD_GRAYSCALE)
            break
        except FileNotFoundError as _:
            _name = str(input(red('could not find file. try again: ')))

    image = image[:, (1920 - 1080) // 2:(1920 - 1080) // 2 + 1080]

    resize_img = cv2.resize(image, (IMAGE_SIZE_X, IMAGE_SIZE_Y))

    cv2.imwrite('tiny.png', resize_img)
    resize_img = np.asarray(resize_img)
    resize_img = [i / 255. for i in resize_img]
    tensor_img = torch.tensor(resize_img, dtype=torch.float).view(-1, 1, IMAGE_SIZE_X, IMAGE_SIZE_Y)
    # cv2.imshow('current test', image)

    tensor_img = tensor_img.to(DEVICE)
    with torch.no_grad():
        output = net(tensor_img)

    argmax = int(torch.argmax(output))
    print(green(OUTPUTS[argmax]), '\tconfidence:', torch.max(output).item())


def main():

    print(yellow('creating network'))
    net = Net()
    print(green('sending network to device'))
    net = net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=.00001)

    train_x, train_y, test_x, test_y = loadTrainingData()
    print(test_y)
    print(blue('begin training'))
    train(net, loss_function, optimizer, train_x, train_y, test_x, test_y)

    # x = train_x + test_x
    # y = train_y + test_y

    # training_data = np.load("hands.npy", allow_pickle=True)
    # x = torch.Tensor([i[0] for i in training_data]).view(-1, IMAGE_SIZE_X, IMAGE_SIZE_Y)
    # x /= 255.  # Change pixel value to range[0, 1)
    # y = [np.argwhere(i[1] == 1.)[0][0] for i in training_data]
    # y = torch.LongTensor(y)
    # acc, loss = test(net, loss_function, optimizer, x, y, size=len(train_x))
    # print(green(f'Total Dataset Accuracy: {acc}, Total Dataset Loss: {loss}'))

    while True:
        choice = str(input('test with a picture filename or type exit: '))
        if choice == 'exit':
            break
        test_image(net, choice)


if __name__ == "__main__":
    main()
