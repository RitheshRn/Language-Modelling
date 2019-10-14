import pickle

with open('reuters_train_trigrams.txt', 'r') as f:
    reuters_train = f.readlines()
reuters_train = [x.strip() for x in reuters_train]

with open('reuters_frequent_trigrams.txt', 'r') as f:
    reuters_val = f.readlines()
reuters_val = [x.strip() for x in reuters_val]

result = []
N = 500
M = 10000
for t in reuters_val[:N]:
    for s in reuters_train:
        if t in s:
            result.append(s)
     
    
if len(result) > M:
    result[:] = result[:M]
    
for i,s in enumerate(result):
    result[i] = s.lower().split(' ')
    
# with open('reuters_sentences_to_append.txt', 'w') as f:
#     for item in result:
#         f.write("%s\n" % str(item))

with open('reuters_sentences_to_append.pkl', 'wb') as f:
    pickle.dump(result, f)
          