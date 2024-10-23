# Report on the Encryption with Neural Networks Code

## 1. Introduction
This report presents a neural network-based approach for text encryption and decryption using PyTorch. The goal is to create a system capable of transforming textual input into an encrypted format and subsequently decrypting it back to its original form. The project employs a combined architecture consisting of an encoder (encryption network) and a decoder (decryption network) trained together to achieve effective information encoding.

## 2. Code Overview

### 2.1 Dependencies
The code utilizes the following Python libraries:
- **PyTorch**: A machine learning library for Python that enables the construction of neural networks.

### 2.2 Data Preparation

**Function: `text_to_integers(text, max_len)`**
- Converts input text into a list of ASCII values corresponding to each character.
- Pads or trims the list to fit a specified maximum length (`max_len`).
- Returns a tensor representation of the ASCII values.

**Function: `scale_data(data)`**
- Normalizes the ASCII values by dividing each value by 255. This scales the values to a range of [0, 1].

**Function: `descale_data(data)`**
- Converts the normalized data back to the ASCII range by multiplying by 255.
- Utilizes `clamp()` to ensure values remain within the valid ASCII range (0 to 255) and rounds them to the nearest integer.

### 2.3 Network Architecture

**Class: `EncryptionNet`**
- Defines the encoder (encryption network) with three fully connected layers:
  - The first layer receives input data, followed by two hidden layers.
  - Sigmoid activation is applied to introduce non-linearity.
  - Outputs encrypted data of specified size.

**Class: `DecryptionNet`**
- Defines the decoder (decryption network) mirroring the structure of the encryption network.
- Converts the encrypted data back into its original format.

**Class: `CryptographyNet`**
- Combines both `EncryptionNet` and `DecryptionNet`.
- Implements the forward pass, which first encrypts the input data and then decrypts it.

### 2.4 Training the Network

**Function: `train_network(net, data, epochs=1000, lr=0.1)`**
- Trains the combined network using mean squared error (MSE) as the loss function and the Adam optimizer.
- Iteratively performs forward passes, computes loss, and updates network weights.
- Logs the training loss every 100 epochs.

### 2.5 Example Usage
The main block of code demonstrates how to:
1. Define a sample input string and its parameters (input size, hidden size, encrypted size).
2. Convert and scale the input data.
3. Initialize and train the cryptography network.
4. Perform encryption and decryption using the trained model.
5. Convert decrypted data back to text and print the results.

### 2.6 Output
The output of the code consists of:
- Encrypted data in normalized form.
- Decrypted data, both normalized and in ASCII values.
- The final decrypted text reconstructed from the ASCII values.

## 3. Results
The neural network successfully learns to encrypt and decrypt the provided text. The training process logs the loss, indicating the network's progress toward minimizing the difference between the original and decrypted data. After training, the decrypted output matches the original input text, demonstrating the network's efficacy.

## 4. Conclusion
This implementation illustrates a simple yet effective method for text encryption and decryption using neural networks. The combined architecture showcases the potential of deep learning in cryptographic applications, although further enhancements could be made to improve robustness, such as increasing complexity or using different architectures. Future work may also explore the application of more advanced techniques, including recurrent neural networks (RNNs) or attention mechanisms, to handle longer sequences of text more effectively.

## 5. References
- PyTorch Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html)
- Neural Networks Overview: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
