from model import ToxicClassifier
import pytorch_lightning as pl
from data_util import *
import argparse
import time
from main_dataloader import JSONDataset, ShuffleDataset, SmallJSONDataset


def coll_fn(data):
    # data format: list of tuples in format (text, label)
    # where text is a string and label is a list of ints/bools (1s and 0s)
    text = np.asarray([tup[0] for tup in data])
    label = torch.from_numpy(np.asarray([np.asarray(tup[1]) for tup in data]))
    return text, label

def main(args):
    # dataset construction
    print("loading {args.dataset} data...")
    t1 = time.time()
    # jigsaw
    # train_dataset = JigsawDataset('data/jigsaw_test.csv')
    # val_dataset = JigsawDataset('data/jigsaw_test.csv')
    # train_dataloader = DataLoader(Subset(train_dataset,range(150)),batch_size=args.batch_size)
    # val_dataloader = DataLoader(Subset(val_dataset,range(150)),batch_size=args.batch_size)
    #
    # # free memory
    # del(train_dataset);del(val_dataset)

    # 4chan
    # the train/val paths given below are for approximately 1/15th of the overall dataset, will update to full dataset when available
    train_path = '/scratch/yx1797/nlp_data/preprocessed_data/train/part-00000/part-00000-56ad4068-8675-445b-9ca4-d6796b1c0f09-c000.json'
    val_path = '/scratch/yx1797/nlp_data/preprocessed_data/val/part-00000/part-00000-22543349-0a64-4c5d-8151-540283a3d07d-c000.json'

    # # use IterableDataset if we are training over the entire dataset, due to memory constraints
    # train_dataset = JSONDataset(train_path, chunkSize=500)
    # train_dataset = ShuffleDataset(train_dataset, buffer_size=500)
    # val_dataset = JSONDataset(val_path, chunkSize=500)
    # val_dataset = ShuffleDataset(val_dataset, buffer_size=500)

    # if using dataset subset, use regular Dataset
    train_dataset = SmallJSONDataset(train_path)
    val_dataset = SmallJSONDataset(val_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True, collate_fn=coll_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, collate_fn=coll_fn)
    elapsed_time = time.time()-t1
    print('Loaded in', elapsed_time, 'seconds, starting training...')
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
    fourchan_test_dataset = SmallJSONDataset('/scratch/yx1797/nlp_data/preprocessed_data/val/part-00015/part-00000-bacfba4e-4789-4205-91bf-98be29e6cbc1-c000.json')
    fourchan_test_dataloader = DataLoader(fourchan_test_dataset, batch_size=args.batch_size, num_workers=2, collate_fn=coll_fn)

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