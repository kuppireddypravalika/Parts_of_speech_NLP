import sys
from collections import defaultdict

trans_counts = defaultdict(lambda: defaultdict(int))
emis_counts = defaultdict(lambda: defaultdict(int))
tag_counts = defaultdict(int)
word_counts = defaultdict(int)
tags = set()
vocab = set()


#-----------------------------------------#

def process_train_data(POS_train):
    global trans_counts, emis_counts, tag_counts, word_counts, tags, vocab
    with open(POS_train, 'r') as file:
        for each_line in file:
            wt_pairs = each_line.strip().split()
            prev_tag = "<s>"
            tag_counts[prev_tag] += 1  
            for wt in wt_pairs:
                if '/' not in wt:
                    #print(f"We find a token wihout tag in the tariing set,token--'{wt}' in line--- {each_line.strip()}")
                    continue
                word, tag = wt.rsplit('/', 1)
                vocab.add(word)
                tags.add(tag)
                trans_counts[prev_tag][tag] += 1
                emis_counts[tag][word] += 1
                tag_counts[tag] += 1
                word_counts[word] += 1
                prev_tag = tag
            trans_counts[prev_tag]["</s>"] += 1  



"""
    ###print statements for debugging

    print("number of starting lines in dataset",tag_counts['<s>'])
    print("count of number of tag NP in data set ",tag_counts['NP'])
    print("count of number of tag . in data set ",tag_counts['.'])
    print("count of number of tag IN in data set ",tag_counts['IN'])

    print("number of emission counts",emis_counts['NP']['Pierre'])
    print("number of emission counts",emis_counts['IN']['of'])

    print("number of transission counts",trans_counts['NP']['NP'])
    print("number of transission counts",trans_counts['.']['</S>'])
    
"""



#-----------------------------------------#

def prob():
    global trans_probs, emis_probs
    trans_probs = cal_trans_prob()
    emis_probs = cal_emis_prob()

def cal_trans_prob():
    trans_probs = defaultdict(lambda: defaultdict(float))
    for x, y in trans_counts.items():
        total_count = sum(y.values())
        for i, j in y.items():
            trans_probs[x][i] = j / total_count
    return trans_probs

def cal_emis_prob():
    emis_probs = defaultdict(lambda: defaultdict(float))
    for x, y in emis_counts.items():
        total_count = sum(y.values())
        for i, j in y.items():
            emis_probs[x][i] = j / total_count
    return emis_probs


#---------------------------------------#

def viterbi_algorithm(sentence):
    global trans_probs, emis_probs
    words = sentence.strip().split()
    W = len(words)
    smoothing_value = 0.000000001  #assigining the small prob for unknown probabilities for smoothing

    V_viterbi = defaultdict(lambda: defaultdict(float))
    back_pointer = defaultdict(lambda: defaultdict(str))

    # Step 1 for the first word
    for tag in tags:
        V_viterbi[tag][0] = (emis_probs[tag].get(words[0], smoothing_value) *
                             trans_probs["<s>"].get(tag, smoothing_value))
        back_pointer[tag][0] = "<s>"

    # Step 2 for remaining words
    for w in range(1, W):
        for tag in tags:
            max_prob = 0
            best_prev_tag = None
            for prev_tag in tags:
                prob = (V_viterbi[prev_tag][w-1] *
                        trans_probs[prev_tag].get(tag, smoothing_value) *
                        emis_probs[tag].get(words[w], smoothing_value))
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
            V_viterbi[tag][w] = max_prob
            back_pointer[tag][w] = best_prev_tag

    # Find best last tag
    max_prob = 0
    best_last_tag = None
    for tag in tags:
        prob = V_viterbi[tag][W-1] * trans_probs[tag].get("</s>", smoothing_value)
        if prob > max_prob:
            max_prob = prob
            best_last_tag = tag

    # Step 3 for backtracking
    best_sequence = [best_last_tag]
    for w in range(W-1, 0, -1):
        best_sequence.append(back_pointer[best_sequence[-1]][w])
    best_sequence.reverse()

    return best_sequence

#----------------------------#

def test_predict(POS_test, output_file):
    total = 0
    correct = 0
    line_num = 0
    mismatches = []

    with open(POS_test, 'r') as test_file, open(output_file, 'w') as out_file:
        for each_line in test_file:
            line_num += 1
            sent = ' '.join([i.rsplit('/', 1)[0] for i in each_line.strip().split()])
            gold_tags = [j.rsplit('/', 1)[1] for j in each_line.strip().split()]
            pred_tags = viterbi_algorithm(sent)
            
            total += len(gold_tags)
            correct += sum([1 for i in range(len(gold_tags)) if gold_tags[i] == pred_tags[i]])

           
            for i, (gold, pred) in enumerate(zip(gold_tags, pred_tags)):
                if gold != pred:
                    word = sent.split()[i]
                    mismatches.append((line_num, word, gold, pred))

            
            out_file.write(' '.join([f"{word}/{tag}" for word, tag in zip(sent.split(), pred_tags)]) + '\n')

 
    acc = (correct / total) * 100
    print(f"Accuracy: {acc:.2f}%")

"""
    if mismatches:
        print("\nwe found the mismatch:")
        for line_num, word, gold, pred in mismatches:
            print(f"Line {line_num}: Word '{word}' - Gold: {gold}, Predicted: {pred}")
    else:
        print("No mismatches.")
"""

#_____________________________________________________#

if __name__ == "__main__":
    while len(sys.argv) != 3:
        print("Please use this command line -'python Viterbi.py POS.train POS.test' or 'python Viterbi.py POS.train.large POS.test'")
        break

    POS_train = sys.argv[1]
    POS_test = sys.argv[2]

    if 'large' in POS_train:
        output_file = "POS.test.large.out"
    else:
        output_file = "POS.test.out"

    process_train_data(POS_train)
    prob()
    test_predict(POS_test, output_file)
