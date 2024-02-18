import torch
import torch.nn as nn
from torchvision import transforms
from AlexNet import create_AlexNet
from DataLoader import LeafDataset, get_dataloader
import time
from WandbLogger import WandbLogger
import pathlib
from tqdm import tqdm
from args_train import get_args_parser
import numpy as np

HEALTHY_CLASSES = [4, 8, 9, 13, 15, 19, 23, 28, 31, 37, 42, 44, 46, 53]

def train(model, train_loader, test_loader, optimizer, criterion, 
    logger, epochs, device, save_freq, save_path):
    
    for epoch in tqdm(range(epochs)):
        
        now = time.time()
        metrics = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'healthy_acc': [], 'precision': [], 'recall': []}
        
        # Train
        model.train()
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            metrics['train_loss'].append(loss.item())
            metrics['train_acc'].append((output.argmax(1) == labels).float().mean().item())

        # Test
        healthy_labels, healthy_preds = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                metrics['test_loss'].append(loss.item())
                metrics['test_acc'].append((output.argmax(1) == labels).float().mean().item())
                
                # get the healthy accuracy
                healthy_classes_tensor = torch.tensor(HEALTHY_CLASSES)
                healthy_labels_mask = (labels.unsqueeze(1) == healthy_classes_tensor).any(dim=1)
                healthy_output_mask = (output.argmax(1).unsqueeze(1) == healthy_classes_tensor).any(dim=1)
                correct_preds = (healthy_labels_mask == healthy_output_mask).sum().item()
                metrics['healthy_acc'].append(correct_preds / len(labels))
                
                # compute precision and recall
                tp = (healthy_labels_mask & healthy_output_mask).sum().item()
                fp = (~healthy_labels_mask & healthy_output_mask).sum().item()
                fn = (healthy_labels_mask & ~healthy_output_mask).sum().item()
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                
                # compute the healthy labels and predictions
                healthy_labels.extend(healthy_labels_mask.int().tolist())
                # apply the softmax function
                output = torch.nn.functional.softmax(output, dim=1)
                # add all the probabilities of the healthy classes
                healthy_preds.extend(output[:, healthy_classes_tensor].sum(dim=1).tolist())
                
        # compute metrics
        for key in metrics:
            metrics[key] = sum(metrics[key]) / len(metrics[key])
        
        # log metrics
        # change healthy_preds shape to shape: (*y_true.shape, num_classes)
        healthy_preds = np.array(healthy_preds)
        healthy_preds = np.stack([1 - healthy_preds, healthy_preds], axis=-1)
        logger.log_roc_curve(healthy_labels, healthy_preds)
        logger.log({
            **metrics,
            'time': time.time() - now
        })

        # print metrics
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {metrics["train_loss"]:.4f}, Test Loss: {metrics["test_loss"]:.4f}, Train Acc: {metrics["train_acc"]:.4f}, Test Acc: {metrics["test_acc"]:.4f}, Healthy Acc: {metrics["healthy_acc"]:.4f}')

        # save model
        if (epoch+1) % save_freq == 0:
            model.save_model(epoch=epoch, optimizer=optimizer, loss=metrics['train_loss'], path=f'{save_path}/model_epoch_{epoch+1}.pt')

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4557, 0.4969, 0.3778], std=[0.1991, 0.1820, 0.2096]),
    ])

    dataset = LeafDataset(args.data_path, transform=transform, load_all_in_memory=False, max_samples=args.max_samples)
    num_classes = dataset.get_nb_classes()
    train_loader, test_loader = get_dataloader(dataset, args.batch_size, args.train_prop)

    # Select configuration
    model = create_AlexNet(num_classes, args.model_size).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.lr)

    # Determining some extra parameters
    args.train_samples = int(args.train_prop * len(dataset))
    args.test_samples = len(dataset) - args.train_samples
    args.n_params = model.n_params

    # Initialize logger
    logger = WandbLogger(config=args, mode=args.wandb_mode)
    
    # Create necessary directories
    complete_save_path = f"{args.save_path}/{args.model_size}"
    pathlib.Path(complete_save_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Training {args.model_size} with {model.n_params} parameters")

    train(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        logger=logger, 
        epochs=args.epochs, 
        device=args.device,
        save_freq=args.save_freq,
        save_path=complete_save_path
    )

    logger.finish()
    