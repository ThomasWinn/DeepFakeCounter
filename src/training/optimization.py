import torch
import torch.nn as nn

def get_loss():
    """
    Get an instance of the CrossEntropyLoss function,
    put it on GPU if it's available
    """
    if torch.cuda.is_available():
        loss = nn.CrossEntropyLoss().cuda()
    else:
        loss = nn.CrossEntropyLoss()
    
    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = 'SGD',
    learning_rate: float = 0.001,
    momentum: float = 0.5,
    weight_decay: float = 0
):
    """Return an instance of our optimizer

    Args:
        model (nn.Module): The model to optimize
        optimizer (str, optional): The optimizer type to use. Defaults to 'SGD'.
        learning_rate (float, optional): learning rate. Defaults to 0.001.
        momentum (float, optional): momenutum for regularization. Defaults to 0.5.
        weight_decay (float, optional): regularization coefficient. Defaults to 0.
    """
    if optimizer.lower() == 'sgd':
        # Create an instance of the SGD
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
    elif optimizer.lower() == "adam":
        # Create an instance of the Adam
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt