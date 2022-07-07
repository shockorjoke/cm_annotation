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
    #print(node_label)
    children_labels = []
    for c in node:
        if isinstance(c, int):
            children_labels.append(c)
            continue
        else:
            children_labels.append(c.label)

    return node_label, parent_label, left_sibling_label, right_sibling_label, children_labels

def extract_rules(rules_tree):

    with open(rules_tree) as f:
        reader = incrementaltreereader(f, strict=True, functions= 'add')

        listed_predecessors = []  #list which save all Mother -> X rules
        self_rules = [] # List which save all Mother -> Node rules
        left_neighbour_rules = [] #List save all (LeftNode, Node)
        right_neighbour_rules = [] #List save all (Node, RighNode)
        listed_node = [] #Listed all non-Terminal Nodes in tree everytime a Node is in a ralation with left/right sibling

        

        for tree, sent, comment in reader:
            for subtree in tree.subtrees():
                # listed_node.append(subtree.label)
                for child in subtree:
                    if isinstance(child, int):
                        self_rules.append((subtree.label,sent[child]))
                        listed_predecessors.append(subtree.label) #This only relevant when we we have big enough data so that the terminal level can also be out in calculation, which not in this case. In this case we does not put the terminal node in the calculation so this line can be put in commend
                        continue
                    else:
                        # cur_len_left=len(left_neighbour_rules)
                        # cur_len_right=len(right_neighbour_rules)
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

def confident_measure():
    listed_predecessors,self_rules,left_neighbour_rules,right_neighbour_rules, listed_node= extract_rules('export_gold_train.txt')

    #prepare counter where all count will be scaled double ##############
    left_neighbour_rules_counter = Counter(left_neighbour_rules) + Counter(left_neighbour_rules)
    right_neighbour_rules_counter = Counter(right_neighbour_rules) + Counter(right_neighbour_rules)
    listed_predecessors_counter = Counter(listed_predecessors) + Counter(listed_predecessors) #each time a node is seen as mother, it counted as 2 times
    self_rules_counter = Counter(self_rules) + Counter(self_rules)
    listed_node_counter = Counter(listed_node) + Counter(listed_node) #each time a node appear in a sentences, it count as 2 times

    # left_neighbour_rules_counter = Counter(left_neighbour_rules) 
    # right_neighbour_rules_counter = Counter(right_neighbour_rules) 
    # listed_predecessors_counter = Counter(listed_predecessors) #each time a node is seen as mother, it counted as 2 times
    # self_rules_counter = Counter(self_rules) 
    # listed_node_counter = Counter(listed_node)  #eac

    
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
    self_rules_posibility = {} #the posibility that the current node is the child its mother node (with laplace smoothing)
    # for self_rule in self_rules:
    for self_rule in self_rules_counter.keys():
        posibility=(self_rules_counter[self_rule])/(listed_predecessors_counter[self_rule[0]]+1)   #add 1 for laplace 1-existent
        self_rules_posibility[self_rule] =  posibility
        
    # for k,v in right_neighbour_rules_posibility.items():
    #     if v>=1:
    #         print(k,v)
        

    return self_rules_posibility,left_neighbour_rules_posibility,right_neighbour_rules_posibility, listed_predecessors_counter, listed_node_counter

#########################################################################################
#NOTE: 'We Donot deal with unseen Node on confident_measure function now, we will deal with it later with if-else on ouput tree'
#   where we can add if Unseen: then probability = 1/(counter + 1)'


#####################################################################################
#Apply  value for input treebank
############################################################################

def output_tree(input_tree):

    #get confident measure databank
    self_rules_posibility,left_neighbour_rules_posibility,right_neighbour_rules_posibility, listed_predecessors_counter, listed_node_counter = confident_measure()
    #print(self_rules_posibility)

    with open('output.bracketed', 'w') as o: #prepare output
        o.close()
    with open(input_tree) as f:
        reader = incrementaltreereader(f, strict=True)
        for tree, sent, comment in reader:
            node_position_dict={} #create a dictionary to save the index position of node and its score
            if tree.label != 'ROOT': #add ROOT token 
                tree = ParentedTree('ROOT', [tree])
            for subtree in tree[0].subtrees():
                node_label, parent_label, left_sibling_label, right_sibling_label, children_labels = extract_features(subtree)
                
                #CASE: when node does not exist in databank then no need to for left/right/children information 
                if listed_node_counter[node_label] == 0: 
                    node_confident = float("{:.4f}".format(math.log(1/((listed_predecessors_counter[parent_label])+1),5))) 
                    #node_confident = math.log(1/((listed_predecessors_counter[parent_label])+1))
                    node_position_dict[subtree.treeposition] = node_confident
                    
                #CASE: when parent node does not exist in databank
                elif listed_node_counter[parent_label]==0: 
                    if isinstance(children_labels[0],int):  #if child node are leafs node then only calculate left/right sibling
                        #unknow in left/right -> posibility = 1/(2*#node + 1)
                        node_confident = math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))),5)+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))),5)
                    else: #calculate left/right/child
                        sum_successor_rule_posibility = 0
                        for c in children_labels:
                            sum_successor_rule_posibility += math.log(self_rules_posibility.get((node_label,c),(1/(listed_predecessors_counter[node_label]+1))),5)
                        mean_successor_rule_posibility = sum_successor_rule_posibility/len(children_labels)
                        node_confident = math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))),5)+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))),5)+mean_successor_rule_posibility
                        node_position_dict[subtree.treeposition] = float("{:.4f}".format(node_confident))
                            
                #CASE: when node, parnet node, left/rightsibling exitst in data but some relation may not
                else:
                    if isinstance(children_labels[0],int): #if child node are leafs node then only calculate self/left/right
                        #unknow in selfnode -> posibility = 1/(2*#mother + 1)
                        node_confident = math.log(self_rules_posibility.get((parent_label,node_label),(1/(listed_predecessors_counter[parent_label]+1))),5)+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))),5)+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))),5)
                        node_position_dict[subtree.treeposition] = float("{:.4f}".format(node_confident))
                        subtree[0]="{}={}".format(subtree[0],sent[subtree[0]])
                    else:
                        sum_successor_rule_posibility = 0
                        for c in children_labels:
                            sum_successor_rule_posibility += math.log(self_rules_posibility.get((node_label,c),(1/(listed_predecessors_counter[node_label]+1))),5)
                        mean_successor_rule_posibility = sum_successor_rule_posibility/len(children_labels)
                        node_confident = math.log(self_rules_posibility.get((parent_label,node_label),(1/(listed_predecessors_counter[parent_label]+1))),5)+math.log(left_neighbour_rules_posibility.get((left_sibling_label,node_label),(1/(listed_node_counter[node_label]+1))),5)+math.log(right_neighbour_rules_posibility.get((node_label,right_sibling_label),(1/(listed_node_counter[node_label]+1))),5)+mean_successor_rule_posibility
                        node_position_dict[subtree.treeposition] = float("{:.4f}".format(node_confident))
            # print(node_position_dict)
            # print('----------------------------------')
            # print(min(node_position_dict.values()))
            for k,v in node_position_dict.items(): #replace node label with score
                tree[0].root[(k)].label = "<{},{}>".format(tree[0].root[(k)].label,v)

            
            ###################################### Scale and replace label with CM, (optinal)
                # subtree.label=node_confident
                # probability_list_to_scale.append(node_confident)

            # for new_subtree in tree[0].subtrees(): #scale the confident between [0,1] and prepare output
            #     if isinstance(new_subtree[0],int):
            #         new_subtree.label = float("{:.4f}".format((new_subtree.label-min(probability_list_to_scale))/(max(probability_list_to_scale)-min(probability_list_to_scale))))
            #         new_subtree[0] = "{}={}".format(new_subtree[0],sent[new_subtree[0]])
            #     else:
            #         new_subtree.label = float("{:.4f}".format((new_subtree.label-min(probability_list_to_scale))/(max(probability_list_to_scale)-min(probability_list_to_scale))))
            #         continue
            ###################################################
            
            with open("output.bracketed", "a") as output: # output tree on output.bracketed
                output.write(str(tree[0])+'\n')

    

    return


if __name__ == '__main__':

    output_tree('input-test-sentences.bracketed')




#############################################################################################
#Modify mit hand jede art von Fehler und testen
#Check Baum für Warnung
#Teste nur mit sentence auf Zeile mit der endung 9 (dev) oder 0 (test)
# (https://gitlab.com/treegrasp/rrgparbank/-/blob/master/partage_parsing/partage_de.bracketed?expanded=true&viewer=simple)
#Gucken ob von 3 wahrscheinlichkeit welche am wictigsten sein kann



# (SENTENCE (CLAUSE (CORE (NP (PRO 0=Das)) (NUC (V 1=gehörte)) 
# (PP (P 2=zu) (NP (OP-DEF 3=den) (CORE_N (AP-PERI (CORE_A (NUC_A (A 4=wirtschaftlichen)))) 
# (NUC_N (N 5=Maßnahmen)) (NP-PERI (OP-DEF 6=der) (CORE_N (AP-PERI (CORE_A (PP (CORE_P (NUC_P (P 7=in)) 
# (NP (CORE_N (NUC_N (N 8=Vorbereitung)))))) (NUC_A (A 9=befindlichen)))) (NUC_N (N 10=Hass-Woche))))))) (. 11=.))))




#correct tree

#fehlt Knoten CLAUSE

#superfluous child nodes with same label CORE -> CORE

#superfluous child nodes with different label 1:

#falsche mutter label NUC anstatt NP

#falsche angehang NP unter PP-PERI anstatt CORE_P


