from model import ToxicClassifier
import pytorch_lightning as pl
from data_util import *
import argparse


def main(args):
    # dataset construction
    print("loading {args.dataset} data...")
    if args.dataset == 'jigsaw':
        dataset = JigsawDataset('data/jigsaw_test.csv')
    elif args.dataset == 'youtube':
        dataset = YoutubeDataset('data/youtube_test.csv')
    elif args.dataset == '4chan':
        pass
    dataloader = DataLoader(dataset,batch_size=1)

    # create and test model
    if args.model_location != None:
        model = ToxicClassifier.load_from_checkpoint(args.model_location)
        pass
    else:
        model = ToxicClassifier()
    trainer = pl.Trainer(max_epochs=1)
    trainer.test(model, dataloader)
    
    return


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description="Evaluate toxic RoBERTa classifier")
    parser.add_argument("--model_location", type=str, default=None, help="Filepath for saved model to evaluate. If none will use Jigsaw pretrained RoBERTa")
    parser.add_argument("--dataset", type=str, default='jigsaw', help="Which data to eval on: options are \"jigsaw\", \"youtube\", \"4chan\"")
    args = parser.parse_args()

    # run 
    main(args)
