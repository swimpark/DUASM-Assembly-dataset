class args_parser:
    """
   Configuration container for training the voxel autoencoder.

   Attributes:
       epochs (int): Number of training epochs.
       iteration (int): Maximum iteration limit (not used in current training loop).
       frac (float): Fraction of data/users used (kept for compatibility).
       local_ep (int): Local update steps (kept for compatibility).
       local_bs (int): Local batch size (kept for compatibility).
       lr (float): Learning rate for optimizer.
       cl_lr (float): Contrastive or auxiliary learning rate (kept for compatibility).
       betas (tuple): Adam optimizer beta parameters.
       model (str): Model type identifier.
       num_channels (int): Number of input channels (1 for voxel grid).
       norm (str): Normalization type placeholder.
       dim (int): Input voxel dimension (e.g., 128 for 128×128×128 grids).
       dataset (str): Dataset identifier.
       optimizer (str): Optimizer type name.
       random_seed (int): Random seed for reproducibility.
       training_size (int): Number of training samples (not enforced inside training script).
       batch_size (int): Batch size used in training.
       loss_record_period (int): Interval for saving loss logs.
   """

    def __init__(self):
        self.epochs = 400
        self.iteration = 1_000_000

        # General training settings
        self.frac = 1
        self.local_ep = 1
        self.local_bs = 1

        # Optimizer settings
        self.lr = 0.00004
        self.cl_lr = 0.00004
        self.betas = (0.8, 0.99)

        # Model configuration
        self.model = "aesnn"
        self.num_channels = 1
        self.norm = "None"
        self.dim = 128

        # Additional metadata
        self.dataset = "DataSet"
        self.optimizer = "Adam"
        self.random_seed = 2
        self.training_size = 800
        self.batch_size = 1
        self.loss_record_period = 1000
