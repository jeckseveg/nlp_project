from model import ToxicClassifier
import pytorch_lightning as pl
from data_util import *
import argparse


def main(args):
    # dataset construction
    print("loading {args.dataset} data...")

    # jigsaw
    train_dataset = JigsawDataset('data/jigsaw_test.csv')
    val_dataset = JigsawDataset('data/jigsaw_test.csv')
    train_dataloader = DataLoader(Subset(train_dataset,range(150)),batch_size=args.batch_size)
    val_dataloader = DataLoader(Subset(val_dataset,range(150)),batch_size=args.batch_size)
    
    # free memory 
    del(train_dataset);del(val_dataset)

    # 4chan
    '''train_dataset = JigsawDataset('data/jigsaw_train.csv')
    val_dataset = JigsawDataset('data/jigsaw_train.csv')
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size)'''

    # create and train model
    model = ToxicClassifier()
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model,val_dataloader,val_dataloader)

    # free memory
    #del(train_dataloader);del(val_dataloader)

    # jigsaw test data
    jigsaw_test_dataset = JigsawDataset('data/jigsaw_test.csv')
    jigsaw_test_dataloader = DataLoader(Subset(jigsaw_test_dataset,range(150)),batch_size=args.batch_size)

    # 4chan test data
    '''fourchan_test_dataset = JigsawDataset('data/jigsaw_test.csv')
    fourchan_test_dataloader = DataLoader(val_dataset,batch_size=args.batch_size)'''

    # youtube test evaluation
    youtube_test_dataset = YoutubeDataset('data/youtube_test.csv')
    youtube_test_dataloader = DataLoader(Subset(youtube_test_dataset,range(150)),batch_size=args.batch_size)

    trainer.test(model, jigsaw_test_dataloader)
    #trainer.test(model, fourchan_test_dataloader)
    trainer.test(model, youtube_test_dataloader)
    
    return


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description="Train toxic RoBERTa classifier on 4chan data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    args = parser.parse_args()

    # run 
    main(args)