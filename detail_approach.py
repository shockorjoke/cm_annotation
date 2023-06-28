from gettext import npgettext
from itertools import count
from pickle import NONE
from platform import node
from discodop.treebank import incrementaltreereader
from discodop.treebank import ParentedTree
from collections import Counter
from discodop.treebank import *
from nbformat import read
from sklearn import neighbors
import math
import itertools
from backup1 import confident_measure
import label_ranks


from soupsieve import comments


############################################################
#Modify input for both databank and treebank
##############################################################

def extract_features(node): #extract features of node: parent, left, right sibling, children
    """Extracts the relevant features from node."""

    node_label = node.label
    parent_label = node.parent.label
    left_sibling_label = None if node.parent_index == 0 else node.parent[node.parent_index - 1].label
    right_sibling_label = None if node.parent_index == len(node.parent) - 1 else node.parent[node.parent_index + 1].label
    
    children_labels = []
    for c in node:
        if isinstance(c, int):
            children_labels.append(c)
            continue
        else:
            children_labels.append(c.label)

    return node_label, parent_label, left_sibling_label, right_sibling_label, children_labels

################ create lists of co-occurence between node and it neighbor
def extract_rules(rules_tree):

    with open(rules_tree) as f:
        reader = incrementaltreereader(f, strict=True, functions= 'add')

        listed_predecessors = []  #list which save all Mother -> X rules
        self_rules = [] # List which save all Mother -> Node rules
        left_neighbour_rules = [] #List save all (LeftNode, Node)
        right_neighbour_rules = [] #List save all (Node, RighNode)
        listed_node = [] #Listed all non-Terminal Nodes in tree everytime a Node is in a ralation with left/right sibling

        

        for tree, sent, comment in reader:
            if tree.label != 'ROOT':
                tree = ParentedTree('ROOT', [tree])
            for subtree in tree.subtrees():                  
                for child in subtree:
                    if isinstance(child, int): #if child is leaf node
                        child = sent[child]
                        self_rules.append((subtree.label,'LEXICAL'))  #add token 'LEXICAL as child node for all leaf node
                        listed_predecessors.append(subtree.label) 
                    else:
                        node_label, parent_label, left_sibling_label, right_sibling_label, children_labels = extract_features(child)
                        listed_predecessors.append(parent_label)
                        self_rules.append((parent_label, node_label))
                        left_neighbour_rules.append((left_sibling_label, node_label))
                        right_neighbour_rules.append((node_label, right_sibling_label))
                        listed_node.append(node_label)

               

    return listed_predecessors,self_rules,left_neighbour_rules,right_neighbour_rules, listed_node

##############################################################


#####################################################################################
#Calculate Confident measure
############################################################################

def events_scores(input_databank):
    listed_predecessors,self_rules,left_neighbour_rules,right_neighbour_rules, listed_node= extract_rules(input_databank)

    ############### prepare counter where all count will be scaled double for scaling methode 
    left_neighbour_rules_counter = Counter(left_neighbour_rules) + Counter(left_neighbour_rules)
    right_neighbour_rules_counter = Counter(right_neighbour_rules) + Counter(right_neighbour_rules)
    listed_predecessors_counter = Counter(listed_predecessors) + Counter(listed_predecessors) #each time a node is seen as mother, it counted as 2 times
    self_rules_counter = Counter(self_rules) + Counter(self_rules)
    listed_node_counter = Counter(listed_node) + Counter(listed_node) #each time a node appear in a sentences, it count as 2 times
    
    
    ################################################################
    
    ################################### approach P(Left Node | Node)
    left_neighbour_rules_posibility= {}  #the posibility that the nodes are sibling
    for left_neighbour_rule in left_neighbour_rules_counter.keys():
        posibility=(left_neighbour_rules_counter[left_neighbour_rule])/(listed_node_counter[left_neighbour_rule[1]]+1)  #add 1 for laplace 1-existent
        left_neighbour_rules_posibility[left_neighbour_rule] = posibility

    #############################################################

    ################################### approach P(Node Right | Node)
    right_neighbour_rules_posibility= {}  #the posibility that the nodes are sibling
    for right_neighbour_rule in right_neighbour_rules_counter.keys():
        posibility=(right_neighbour_rules_counter[right_neighbour_rule])/(listed_node_counter[right_neighbour_rule[0]]+1)   #add 1 for laplace 1-existent
        right_neighbour_rules_posibility[right_neighbour_rule] = posibility
    ############################################################
    
    ################################### P(Node|Mother)
    self_rules_posibility = {} #the posibility that the current node is the child its mother node 
    # for self_rule in self_rules:
    for self_rule in self_rules_counter.keys():
        posibility=(self_rules_counter[self_rule])/(listed_predecessors_counter[self_rule[0]]+1)   #add 1 for laplace 1-existent
        self_rules_posibility[self_rule] =  posibility

    return self_rules_posibility,left_neighbour_rules_posibility,right_neighbour_rules_posibility, listed_predecessors_counter, listed_node_counter

#########################################################################################
#NOTE: 'We Donot deal with unseen Node on confident_measure function now, we will deal with it later with if-else on ouput tree'
#   where we can add if Unseen: then probability = 1/(counter + 1)'


#####################################################################################
#Apply  value for input treebank
############################################################################

def output_tree(input_tree):

    #get confident measure databank
    self_rules_posibility,left_neighbour_rules_posibility,right_neighbour_rules_posibility, listed_predecessors_counter, listed_node_counter = events_scores('brackets_export_gold_train.txt')

    confident_node_label_list_dict=label_ranks.get_confident_label_list('brackets_export_gold_train.txt')

    # create text file to save output sentences, error sentences found by absolute compare and bottom limit compare
    with open('output.bracketed', 'w') as o, open('absolut_error_sentences.txt', "w") as asolut, open('bottom_limit_error_sentences.txt', "w") as range: #prepare output
        o.close()
        asolut.close()
        range.close()
    with open(input_tree) as f, open('absolut_error_sentences.txt', "a") as asolut, open('bottom_limit_error_sentences.txt', "a") as range:
        absolut_error_sentences =[]
        bottom_limit_error_sentences = []
        
        reader = incrementaltreereader(f, strict=True, functions= 'add')
        
        for tree, sent, comment in reader:
            unknow_node = absolute_error = bottom_limit_error = False
            posible_error_positions = set()
            node_position_dict={} #create a dictionary to save the index position of node and its score
            if tree.label != 'ROOT': #add ROOT token 
                tree = ParentedTree('ROOT', [tree])
            
            for subtree in tree[0].subtrees():
                node_label, parent_label, left_sibling_label, right_sibling_label, children_labels = extract_features(subtree)
                
                #CASE: when node does not exist in databank 
                if listed_node_counter[node_label] == 0: 
                    node_confident = float("{:.4f}".format(4*math.log(1/((listed_predecessors_counter[parent_label])+1)))) 
                    node_position_dict[subtree.treeposition] = node_confident
                    unknow_node = True
                    posible_error_positions.add('UNK <'+node_label+ '> '+str(float("{:.4f}".format(node_confident))))
                #CASE: when parent node does not exist in databank
                elif listed_predecessors_counter[parent_label]==0: 
                    if isinstance(children_labels[0],int):  #if child node are leafs node -> LEXICAL child 
                        node_confident = math.log(1/(listed_node_counter[node_label]+1))+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))))+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))))+math.log(self_rules_posibility.get((node_label,'LEXICAL'),(1/(listed_predecessors_counter[node_label]+1))))
                        node_position_dict[subtree.treeposition] = float("{:.4f}".format(node_confident))
                    else: #calculate left/right/child + special case for parent node = 1/counter(node) 
                        sum_successor_rule_posibility = 0
                        for c in children_labels:
                            sum_successor_rule_posibility += math.log(self_rules_posibility.get((node_label,c),(1/(listed_predecessors_counter[node_label]+1))))
                        mean_successor_rule_posibility = sum_successor_rule_posibility/len(children_labels)
                        node_confident = math.log(1/(listed_node_counter[node_label]+1))+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))))+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))))+mean_successor_rule_posibility
                        node_position_dict[subtree.treeposition] = float("{:.4f}".format(node_confident))

                #CASE: when node, parent node, left/rightsibling exitst in data but some relation may not
                else:
                    if isinstance(children_labels[0],int): #if child node are leafs node  -> LEXICAL child
                        #unknow in selfnode -> posibility = 1/(2*#mother + 1)
                        node_confident = math.log(self_rules_posibility.get((parent_label,node_label),(1/(listed_predecessors_counter[parent_label]+1))))+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))))+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))))+math.log(self_rules_posibility.get((node_label,'LEXICAL'),(1/(listed_predecessors_counter[node_label]+1))))
                        node_position_dict[subtree.treeposition] = float("{:.4f}".format(node_confident))
                        #sent[subtree[0]] = "{}={}".format(subtree[0],sent[subtree[0]])
                        subtree[0]="{}={}".format(subtree[0],sent[subtree[0]])
                    else:
                        sum_successor_rule_posibility = 0
                        for c in children_labels:
                            sum_successor_rule_posibility += math.log(self_rules_posibility.get((node_label,c),(1/(listed_predecessors_counter[node_label]+1))))
                        mean_successor_rule_posibility = sum_successor_rule_posibility/len(children_labels)
                        node_confident = math.log(self_rules_posibility.get((parent_label,node_label),(1/(listed_predecessors_counter[parent_label]+1))))+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))))+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))))+mean_successor_rule_posibility
                        node_position_dict[subtree.treeposition] = float("{:.4f}".format(node_confident))
                
                #error detect switch 
                if unknow_node == False:
                    if float("{:.4f}".format(node_confident)) not in confident_node_label_list_dict[node_label]:
                        if float("{:.4f}".format(node_confident)) < min(confident_node_label_list_dict[node_label]):
                            posible_error_positions.add('STLo <'+node_label + '> '+str(float("{:.4f}".format(node_confident))))
                        else:
                            posible_error_positions.add('SNII <'+node_label+ '> '+str(float("{:.4f}".format(node_confident))))
                        absolute_error = True
                if unknow_node == False:
                    if unknow_node == False and float("{:.4f}".format(node_confident)) < min(confident_node_label_list_dict[node_label]) or float("{:.4f}".format(node_confident)) > max(confident_node_label_list_dict[node_label]):
                        if float("{:.4f}".format(node_confident)) < min(confident_node_label_list_dict[node_label]):
                            posible_error_positions.add('STLo <'+node_label + '> '+str(float("{:.4f}".format(node_confident))))
                        if float("{:.4f}".format(node_confident)) > max(confident_node_label_list_dict[node_label]):
                            posible_error_positions.add('STLa <'+node_label+ '> '+str(float("{:.4f}".format(node_confident))))
                        bottom_limit_error = True

            # save tree in case node unknow  
            if unknow_node == True:
                absolute_error = False
                bottom_limit_error = False
                absolut_error_sentences.append(tree)
                bottom_limit_error_sentences.append(tree)
                asolut.write(str(tree[0])+'\n')
                asolut.write(str(posible_error_positions)+'\n')
                range.write(str(tree[0])+'\n')
                range.write(str(posible_error_positions)+'\n')
                unknow_node=False
            #save tree in case error found by absolute compare
            if absolute_error == True:
                absolut_error_sentences.append(tree)
                asolut.write(str(tree[0])+'\n')
                asolut.write(str(posible_error_positions)+'\n')
                absolute_error = False
            #save tree in case error found by bottom limit compare
            if bottom_limit_error == True:
                bottom_limit_error_sentences.append(tree)
                range.write(str(tree[0])+'\n')
                range.write(str(posible_error_positions)+'\n')
                bottom_limit_error = False

                
            for k,v in node_position_dict.items(): #replace node label with score
                tree[0].root[(k)].label = "<{},{}>".format(tree[0].root[(k)].label,v)

            
            with open("output.bracketed", "a") as output: # output tree on output.bracketed
                output.write(str(tree[0])+'\n')
        # with open ('test1.txt', 'a') as fd: 
        #     fd.write(str((len(absolut_error_sentences),len(bottom_limit_error_sentences))))
    

    return

###############################################commen thhose line if using tempcode.py
if __name__ == '__main__':

        
    output_tree('partage_de.bracketed')  #commen this line if using tempcode.py
#     output_tree('brackets_export_gold_dev.txt')
#############################################################





