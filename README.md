# CNN_Hand_Emoji

The requirements of the project are `numpy`, `torch`, and `curtsies`.

The datasets to train the project are located here. `hands_25.npy` contains all 25000 images (25 emojis at 1000 images each). `hands_10.npy` contains the first 10000 images, noted by the commented `OUTPUTS` list at the beginning of `cnn.py`.

After training, test the network with an image in the working directory.

DISCLAIMER: `hands_25.npy` causes issues with the network's ouput when providing images beyond `love-you_gesture` in the output list. I will debug this soon.

To use `hands_10.npy` (or `hands_25.npy`), verify two things:
1) The output of the final linear layer of the network needs to be changed to `10` (or `25` respectively).
2) The dataset filename needs to be changed appropriately. The dataset is loaded one time at the head of `loadTrainingData`.

To run the script, type `python cnn.py`.

The actual dataset (1080p PNGs) will be uploaded to the save drive link provided above soon (it needs remaking for more varied training data). I will look into publishing the dataset as well.
