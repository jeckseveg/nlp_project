from model import ToxicClassifier
import pytorch_lightning as pl
from data_util import *
import argparse
from main_dataloader import JSONDataset, ShuffleDataset


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
    # the train/val paths given below are for approximately 1/15th of the overall dataset, will update to full dataset when available
    train_path = '/scratch/yx1797/nlp_data/preprocessed_data/train/part-00000/part-00000-56ad4068-8675-445b-9ca4-d6796b1c0f09-c000.json'
    val_path = '/scratch/yx1797/nlp_data/preprocessed_data/val/part-00000/part-00000-22543349-0a64-4c5d-8151-540283a3d07d-c000.json'
    train_dataset = JSONDataset(train_path, chunkSize=1000)
    train_dataset = ShuffleDataset(train_dataset, buffer_size=1024)
    val_dataset = JSONDataset(val_path, chunkSize=1000)
    val_dataset = ShuffleDataset(val_dataset, buffer_size=1024)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # create and train model
    model = ToxicClassifier()
    trainer = pl.Trainer(max_epochs=args.epochs)
    trainer.fit(model,train_dataloader,val_dataloader)

    # free memory
    #del(train_dataloader);del(val_dataloader)

    # jigsaw test data
    # jigsaw_test_dataset = JigsawDataset('data/jigsaw_test.csv')
    # jigsaw_test_dataloader = DataLoader(Subset(jigsaw_test_dataset,range(150)),batch_size=args.batch_size)

    # 4chan test data
    fourchan_test_dataset = JSONDataset('/scratch/yx1797/nlp_data/preprocessed_data/val/part-00001/part-00000-4c419c7a-be9d-460c-8e35-46326dd66922-c000.json', chunkSize=1000)
    fourchan_test_dataloader = DataLoader(fourchan_test_dataset, batch_size=args.batch_size)

    # youtube test evaluation
    # youtube_test_dataset = YoutubeDataset('data/youtube_test.csv')
    # youtube_test_dataloader = DataLoader(Subset(youtube_test_dataset,range(150)),batch_size=args.batch_size)

    # trainer.test(model, jigsaw_test_dataloader)
    trainer.test(model, fourchan_test_dataloader)
    # trainer.test(model, youtube_test_dataloader)
    
    return


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description="Train toxic RoBERTa classifier on 4chan data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data loader")
    args = parser.parse_args()

    # run 
    main(args)