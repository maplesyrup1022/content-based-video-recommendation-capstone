import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import evaluate
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=100, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
parser.add_argument("--pretrained_embedding", 
	type=str,
	default=None,  
	help="if there is pretrained embedding")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
test_num = 1682
train_data, test_data, user_num ,item_num, train_mat = data_utils.load_all()

# construct the train and test datasets
train_dataset = data_utils.NCFData(
		train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
		test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
		batch_size=args.test_num_ng, shuffle=False, num_workers=0)
pretrained_item_embedding = None
if args.pretrained_embedding is not None:
	with open(args.pretrained_embedding, 'rb') as f:
		pretrained_item_embedding = np.load(f, allow_pickle=True)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, pretrained_item_embedding, GMF_model, MLP_model)
if args.gpu != '':
	print('not using gpu')
	model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

# writer = SummaryWriter() # for visualization

########################### TRAINING #####################################
count, best_hr = 0, 0
films_watched = []
with open(config.films_watched, 'r') as fd:
	line = fd.readline()
	while line != None and line != '':
		arr = line.split('\t')
		films_watched.append([int(i) for i in arr])
		line = fd.readline()
films = pd.read_csv('items_with_youtube_url_final_ver.csv')
directors_gender = films[['movie id', 'director_gender']]
actors_gender = films[['movie id', 'female_actors_exist', 'female_actors_cutoff']]

for epoch in range(args.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()
	train_loader.dataset.ng_sample()

	for user, item, label in train_loader:
		label = label.float()
		if args.gpu != '':
			user = user.cuda()
			item = item.cuda()
			label = label.cuda()

		model.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		count += 1

	model.eval()
	HR, NDCG, exposure_rate, exposure_index_against_base, female_exist_er, gender_balanced_er = \
		evaluate.metrics(model, args.gpu, test_loader, args.top_k,\
			 films_watched, directors_gender, actors_gender)

	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}\tDirector Exposure Rate: {:.3f}\t\
		Director Exposure Index: {:.3f}\tFemale Exist Exposure \
		Rate: {:.3f}\tGender Balanced Exposure Rate: {:.3f}".format(np.mean(HR),\
		 np.mean(NDCG), np.mean(exposure_rate), np.mean(exposure_index_against_base),\
		 np.mean(female_exist_er), np.mean(gender_balanced_er)))

	if HR > best_hr:
		best_hr, best_ndcg, best_epoch, best_exposure_rate, best_exposure_index_against_base,\
		 best_exist_er, best_balanced_er = HR, NDCG, epoch, exposure_rate, \
			exposure_index_against_base, female_exist_er, gender_balanced_er
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model, 
				'{}{}.pth'.format(config.model_path, config.model))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}, Director Exposure Rate = {:.3f},\
	 Director Exposure Index = {:.3f}, Female Exist Exposure Rate = {:.3f},\
	Gender Balanced Exposure Rate = {:.3f}".format(best_epoch, best_hr, best_ndcg, \
		best_exposure_rate, best_exposure_index_against_base, best_exist_er, best_balanced_er))
