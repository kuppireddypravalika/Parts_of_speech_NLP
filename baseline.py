import sys
from collections import Counter, defaultdict

def fn_process_train_data(POS_train):
   
    word_given_tag_count = defaultdict(Counter)
    only_tag_counts = Counter()
    with open(POS_train, 'r') as f_train:
        for each_line in f_train:
            tokens = each_line.strip().split()
            for each_token in tokens:
                if '/' not in each_token:
                    #print(f"We found the word with no tag - '{each_token}' in line: {each_line.strip()}")
                    continue  # this if function is to ignore the words that are not tagged in the training set
                W, T = each_token.rsplit('/', 1)
                word_given_tag_count[W][T] += 1
                only_tag_counts[T] += 1

    frequent_tag = {}
    for word, counts in word_given_tag_count.items():
        common_tag, _ = counts.most_common(1)[0]
        frequent_tag[word] = common_tag
    common_tag = only_tag_counts.most_common(1)[0][0]

    #print("This is the most common tag seen in the training  dataset:", common_tag)
    
    return frequent_tag, common_tag


    #--------------------------------------------------#

def fn_test_data(POS_test, frequent_tag, common_tag):
    correct = 0
    total = 0

    with open(POS_test, 'r') as f_test:
        for each_line in f_test:
            tokens = each_line.strip().split()

            for token in tokens:
                if '/' not in token:
                    #print(f"We found the word with no tag - '{token}' in line: {each_line.strip()}")
                    continue
                
                word, gold_tag = token.rsplit('/', 1)
                
                predicted_tag = frequent_tag.get(word, common_tag)
                
        
                if predicted_tag == gold_tag:
                    correct += 1
                total += 1


    if total > 0:
        accuracy = (correct / total) * 100
    else:
        accuracy = 0.0

    print(f"Accuracy: {accuracy:.2f}%")


    #--------------------------------------------------#

if __name__ == "__main__":
    while len(sys.argv) != 3:
        print("Please use this command line -- 'python baseline.py POS.train POS.test'")
        break
 
    POS_train = sys.argv[1]
    POS_test = sys.argv[2]

    # with the help of this funtion - fn_process_train_data. we can count most frequent tags and most common tag occured in training data set
    frequent_tag, common_tag = fn_process_train_data(POS_train)
    
    # with the help of this funtion - fn_test_data. we can assign most frequent tags and most common tag for new word occured in test data
    fn_test_data(POS_test, frequent_tag, common_tag)
