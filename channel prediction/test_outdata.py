import pickle

with open('channel prediction/datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

test_features = datasets['test_features']
test_targets = datasets['test_targets']
final_nmse = 0  

for i in range(len(test_targets)):
        nmse = ((test_features[i][0,0] - test_targets[i]) ** 2) /  (test_targets[i] ** 2)
        final_nmse += nmse
final_nmse = final_nmse / len(test_targets) 
print(f'Normalized Mean Squared Error (NMSE) on the test set: {final_nmse:.8f}')