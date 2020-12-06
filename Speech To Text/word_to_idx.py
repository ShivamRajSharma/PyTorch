import pickle
char_to_idx = {}
idx_to_char = {}
f = open('vocab.txt').read().strip().split('\n')
for char_label in f:
    char, label = char_label.split()
    idx_to_char[int(label)] = char
    char_to_idx[char] = int(label) 

pickle.dump(char_to_idx, open('input/char_to_idx.pickle', 'wb'))
pickle.dump(idx_to_char, open('input/idx_to_char.pickle', 'wb'))
