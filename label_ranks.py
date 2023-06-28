from gettext import npgettext
from itertools import count
from pickle import NONE
from discodop.treebank import incrementaltreereader
from discodop.treebank import ParentedTree
from collections import Counter
from discodop.treebank import *
from nbformat import read
from sklearn import neighbors
import math
import detail_approach

from soupsieve import comments


def get_confident_label_list(input_tree):
    
    #get confident measure databank
    self_rules_posibility,left_neighbour_rules_posibility,right_neighbour_rules_posibility, listed_predecessors_counter, listed_node_counter = detail_approach.events_scores('brackets_export_gold_train.txt')
    with open(input_tree) as f, open("confident_label_list.text", "w") as output1:
        reader = incrementaltreereader(f, strict=True,functions= 'add')
        confident_node_label_list_dict={}
        for tree, sent, comment in reader:

             #create a dictionary to save the index position of node and its score
            if tree.label != 'ROOT': #add ROOT token 
                tree = ParentedTree('ROOT', [tree])
            for subtree in tree[0].subtrees():
                node_label, parent_label, left_sibling_label, right_sibling_label, children_labels = detail_approach.extract_features(subtree)
                if isinstance(children_labels[0],int): 
                    node_confident = math.log(self_rules_posibility.get((parent_label,node_label)))+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label)))+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label)))+math.log(self_rules_posibility.get((node_label,'LEXICAL')))
                    confident_node_label_list_dict.setdefault(node_label,[float("{:.4f}".format(node_confident))]).append(float("{:.4f}".format(node_confident)))
                    subtree[0]="{}={}".format(subtree[0],sent[subtree[0]])
                    
                else:
                    sum_successor_rule_posibility = 0
                    for c in children_labels:
                        sum_successor_rule_posibility += math.log(self_rules_posibility.get((node_label,c)))
                    mean_successor_rule_posibility = sum_successor_rule_posibility/len(children_labels)
                    node_confident = math.log(self_rules_posibility.get((parent_label,node_label)))+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label)))+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label)))+mean_successor_rule_posibility
                    confident_node_label_list_dict.setdefault(node_label,[float("{:.4f}".format(node_confident))]).append(float("{:.4f}".format(node_confident)))

        output1.write(str(confident_node_label_list_dict)+'\n') #write confident list on text data

    return confident_node_label_list_dict



if __name__ == '__main__':
    
    get_confident_label_list('brackets_export_gold_train.txt')


      
      
      