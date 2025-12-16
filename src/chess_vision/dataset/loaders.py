from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from chess_vision.dataset.dataset import SquareDataset

def load_dataset():
    data_dir = 'data/'
    image_size = 224
    batch_size = 32
    val_split = 0.2

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # upsampling to match pretrained model
        transforms.ToTensor(),
        transforms.Normalize( # Normalizing to match pretrained model
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = SquareDataset(data_dir, transform)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Loaded Data")
    return train_loader, val_loader

