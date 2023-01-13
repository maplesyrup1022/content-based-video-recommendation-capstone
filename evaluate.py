import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0

#read in the director gender
def female_director_exposure_rate(directors_gender, pred_items):
	#how many of the work recommended are by female directors?
	female_director_exposure_rate = 0
	unclear = 0
	for item in pred_items:
		if directors_gender.loc[item, 'director_gender'] == 'F':
			female_director_exposure_rate += 1
		if directors_gender.loc[item, 'director_gender'] == 'Unclear':
			unclear += 1
	female_director_exposure_rate /= (len(pred_items)-unclear)

	#percentage of female directors in the dataset, used to calculate the index
	base_female_director_exposure_rate = len(directors_gender[directors_gender['director_gender'] == 'F'])\
		/(len(directors_gender) - len(directors_gender[directors_gender['director_gender'] == 'Unclear']))
	
	return female_director_exposure_rate, female_director_exposure_rate-base_female_director_exposure_rate

def female_actor_exposure_rate(actors_gender, pred_items):
	#exposure rate of female actors exist films
	female_exist_er = 0
	unclear = 0

	#exposure rate of relatively gender balanced films
	gender_balanced_er = 0

	for item in pred_items:
		if actors_gender.loc[item, 'female_actors_exist'] == 1:
			female_exist_er += 1
		if str(actors_gender.loc[item, 'female_actors_cutoff']) == '1':
			gender_balanced_er += 1
		if actors_gender.loc[item, 'female_actors_cutoff'] == 'Unclear':
			unclear += 1
	
	female_exist_er /= (len(pred_items)-unclear)
	gender_balanced_er /= (len(pred_items)-unclear)
	#all_unclear = len(actors_gender[actors_gender['female_actors_cutoff'] == 'Unclear'])

	return female_exist_er, gender_balanced_er

#metrics is called for once in the testing process
def metrics(model, gpu, test_loader, top_k, films_watched, directors_gender, actors_gender):
	#films_watched is a list of films user has watched
	HR, NDCG, exposure_rate, exposure_index_against_base, female_exist_er, gender_balanced_er= [], [], [], [], [], []

	for user, item, label in test_loader:
		if gpu != '':
			user = user.cuda()
			item = item.cuda()
		#item is a tensor of all of the test movie id

		predictions = model(user, item)
		#set those films where user have watched to negative probability
		for i in range(len(films_watched)):
			if films_watched[i][0] == user[0]:
				for j in range(1, len(films_watched[i])):
					index = (item == films_watched[i][j]).nonzero(as_tuple=True)[0]
					predictions[index] = -1

		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()
		#print(recommends)
		#with different size of k, how will the ration change, mitigate gender bias with changed k, different position of k??

		gt_item = item[0].item()
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

		#get exposure rate and index for this recommendation
		ER, EI = female_director_exposure_rate(directors_gender, recommends)
		exposure_rate.append(ER)
		exposure_index_against_base.append(EI)

		exist_ER, balanced_ER = female_actor_exposure_rate(actors_gender, recommends)
		female_exist_er.append(exist_ER)
		gender_balanced_er.append(balanced_ER)
		
	return np.mean(HR), np.mean(NDCG), np.mean(exposure_rate), \
		np.mean(exposure_index_against_base), np.mean(female_exist_er), np.mean(gender_balanced_er)
