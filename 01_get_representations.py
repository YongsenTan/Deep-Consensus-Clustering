import argparse
import os
import random
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Model
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Consensus Clustering')

    # Path parameters
    parser.add_argument('--input_path', type=str, default='./dataset',
                        help='path of input dataset')
    parser.add_argument('--filename_train', type=str, default='data_train.pkl',
                        help='path of the data train')
    parser.add_argument('--filename_valid',  type=str, default='data_valid.pkl',
                        help='path of the data valid')
    parser.add_argument('--filename_test', type=str, default='data_valid.pkl',
                        help='path of the data test')
    parser.add_argument('--filename_data', type=str, default='data.pkl',
                        help='path of the whole data')
    parser.add_argument('--log_path', type=str, default='./runs',
                        help='path of the tensorboard log')

    # Training parameters
    parser.add_argument('--pre_epoch', type=int, default=1,
                        help='number of pre-train epochs')
    parser.add_argument('--epoch', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout in LSTM')

    # Model parameters
    parser.add_argument('--input_dim', type=int, required=True,
                        help='number of original input feature size')
    parser.add_argument('--hidden_dims', type=int, required=True,
                        help='latent space dimension')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='number of hidden size in LSTM')
    parser.add_argument('--lambda_AE', type=float, default=1.0,
                        help='lambda of AE in iteration')
    parser.add_argument('--lambda_outcome', type=float, default=10.0,
                        help='lambda of outcome in iteration')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    # Utility parameters
    parser.add_argument('--cuda', type=int, default=1,
                        help='If use cuda')
    args = parser.parse_args()
    return args


def main(args):
    # Set the random seed manually for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load data
    data_train = AKIData(args.input_path + args.filename_train)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)
    data_valid = AKIData(args.input_path + args.filename_valid)
    dataloader_valid = DataLoader(data_valid, batch_size=args.batch_size, shuffle=True)

    # mkdir
    args.output_path = "./representations"
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # tensorboard writer
    writer = SummaryWriter(args.log_path)

    model = Model(args)
    print(model)

    # pretrain
    model.pretrain(dataloader_train)

    saved_iter = -1
    saved_iter_list = []
    min_outcome_likelihood = np.inf

    for e in range(args.epoch):
        print("=======================================")
        print("epoch:", e)

        # train
        train_ae_loss, train_outcome_loss, train_outcome_auc_score = model.fit(dataloader_train)

        # test on validation set
        test_ae_loss, test_outcome_loss, test_outcome_auc_score = evaluate(model, dataloader_valid)

        writer.add_scalar('train_ae_loss', train_ae_loss, e)
        writer.add_scalar('validation_ae_loss', test_ae_loss, e)
        writer.add_scalar('train_outcome_loss', train_outcome_loss, e)
        writer.add_scalar('validation_outcome_loss', train_outcome_loss, e)
        writer.add_scalar('train_outcome_auc', train_outcome_auc_score, e)
        writer.add_scalar('validation_outcome_auc', test_outcome_auc_score, e)

        # save model
        if test_outcome_loss < min_outcome_likelihood:
            min_outcome_likelihood = test_outcome_loss

            torch.save(model.state_dict(), args.output_path + '/model_best_' + str(args.hidden_dims) + '.pt')
            print("model saved")

            saved_iter_list.append(e)
            saved_iter = e

    print("=======================================")
    print("saved_iter_list=", saved_iter_list)
    print("saved_iter = ", saved_iter)

    data = AKIData(args.input_path + args.filename_data)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    model_best = Model(args)
    model_best.load_state_dict(torch.load(args.output_path + '/model_best_' + str(args.hidden_dims) + '.pt',
                                          map_location=torch.device('cuda' if args.cuda else 'cpu')))

    # test on test set
    data_test = AKIData(args.input_path + args.filename_test)
    dataloader_test = DataLoader(data_test, batch_size=16, shuffle=False)
    test_ae_loss, test_outcome_loss, test_outcome_auc_score = evaluate(model_best, dataloader_test)
    print('test_ae_loss: %.3f' % test_ae_loss)
    print('test_outcome_loss: %.3f' % test_outcome_loss)
    print('test_outcome_auc_score: %.3f' % test_outcome_auc_score)

    # save representation
    rep = get_embedding(model_best, dataloader)
    f = open(args.output_path + '/rep_' + str(args.hidden_dims) + '.pkl', 'wb')
    pickle.dump(rep, f)
    f.close()


if __name__ == '__main__':
    args_main = parse_args()
    main(args_main)
