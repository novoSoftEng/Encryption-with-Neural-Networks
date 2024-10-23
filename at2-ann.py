import torch
import torch.nn as nn
import torch.optim as optim

# Function to convert text to integers (based on character ASCII values)
def text_to_integers(text, max_len):
    # Get ASCII values of each character in the text
    ascii_vals = [ord(c) for c in text]

    # Pad or trim to fit the required length (max_len)
    if len(ascii_vals) < max_len:
        ascii_vals += [0] * (max_len - len(ascii_vals))  # Padding with zeros if too short
    else:
        ascii_vals = ascii_vals[:max_len]  # Trimming if too long

    return torch.tensor(ascii_vals, dtype=torch.float32)

# Scale function (normalization to [0, 1])
def scale_data(data):
    return data / 255.0  # Scale ASCII values (0 to 255) to [0, 1]

# Descale function (denormalization)
def descale_data(data):
    return (data * 255.0).clamp(0, 255).round()  # Scale back to [0, 255] and round

# Encryption network (Encoder)
class EncryptionNet(nn.Module):
    def __init__(self, input_size, hidden_size, encrypted_size):
        super(EncryptionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, encrypted_size)
        self.activation = nn.Sigmoid()  # Activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        encrypted = self.fc3(x)
        return encrypted

# Decryption network (Decoder)
class DecryptionNet(nn.Module):
    def __init__(self, encrypted_size, hidden_size, output_size):
        super(DecryptionNet, self).__init__()
        self.fc1 = nn.Linear(encrypted_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, encrypted):
        x = self.activation(self.fc1(encrypted))
        x = self.fc2(x)
        decrypted = self.fc3(x)
        return decrypted

# Combined Network (for training encryption and decryption together)
class CryptographyNet(nn.Module):
    def __init__(self, input_size, hidden_size, encrypted_size):
        super(CryptographyNet, self).__init__()
        self.encryption = EncryptionNet(input_size, hidden_size, encrypted_size)
        self.decryption = DecryptionNet(encrypted_size, hidden_size, input_size)

    def forward(self, x):
        encrypted = self.encryption(x)
        decrypted = self.decryption(encrypted)
        return decrypted

# Training function
def train_network(net, data, epochs=1000, lr=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = net(data)
        
        # Compute loss (mean squared error between original and decrypted data)
        loss = criterion(output, data)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    print("Training finished!")

# Example usage
if __name__ == '__main__':
    # Sample text
    sentence = "something is wrong"
    input_size = len(sentence)  # Max length of input text (characters)
    hidden_size = 64  # Size of hidden layers
    encrypted_size = 1  # Size of ciphertext (compressed size)
    
    # Convert the text into integers (ASCII values)
    data = text_to_integers(sentence, input_size).unsqueeze(0)  # Add batch dimension
    
    # Scale the input data
    data_scaled = scale_data(data)

    # Initialize the cryptography network
    crypto_net = CryptographyNet(input_size, hidden_size, encrypted_size)

    # Train the combined network
    train_network(crypto_net, data_scaled, epochs=100, lr=0.01)

    # After training, we can use the encryption and decryption networks independently:
    
    # Encryption process
    with torch.no_grad():
        encrypted_data = crypto_net.encryption(data_scaled)
        print("Encrypted data:", encrypted_data)

        # Decryption process
        decrypted_data = crypto_net.decryption(encrypted_data)
        print("Decrypted data (normalized):", decrypted_data)

        # Descale the decrypted data
        decrypted_data_descale = descale_data(decrypted_data)
        print("Decrypted data (ASCII):", decrypted_data_descale)

    # Convert decrypted data back to text (ASCII to characters)
    decrypted_text = ''.join([chr(int(x)) for x in decrypted_data_descale.squeeze().tolist() if x > 0])
    print("Decrypted text:", decrypted_text)
