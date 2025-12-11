# Model hyperparameters
# For Nodes
in_channels = 128  # Node Embedding vector size
out_channels = in_channels  # Output dimension should match input for reconstruction

#For GAE
latent_dim = 16  # Latent embedding size

# Training settings
learning_rate = 0.00001  # Learning rate for the optimizer
epochs = 300  # Number of training epochs
batch_size = 1  # Batch size for training
