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
import detail_approach
import pandas as pd
import random


from torch import NoneType

###################################################### For bracket format input test
# file = open('partage_de.bracketed')
  
# # read the content of the file opened
# content = file.readlines()
  

# with open("test-sentences.bracketed", "w") as output: # output tree on output.bracketed
#     output.write(content[18])
#     output.write(content[48])
#     output.write(content[68])
#     output.write(content[128])
#     output.write(content[158])
#     output.write(content[188])
#     output.write(content[198])
#     output.write(content[208])
#     output.write(content[268])
#     output.write(content[338])
    

#####################################################################

with open("input-test-sentences.bracketed", "w") as output: #prepare input
    output.close()
    
    
# with open('brackets_testsen.txt') as f: #edit sentence for input in bracket Format
with open('brackets_export_gold_dev.txt') as f: #edit sentence for input in bracket Format

    reader = incrementaltreereader(f, strict=True, functions= 'add')
    for tree, sent, comment in reader:

        tree_position = [] # position-list of node
        right_most_child_position = [] #position-list of right most child with left sibling
        #left_most_child_position = []
        for subtree in tree.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0]="{}={}".format(subtree[0],sent[subtree[0]])

                else:
                    tree_position.append(subtree.treeposition)
                    if child.right_sibling == None and child.left_sibling != None:
                        right_most_child_position.append(subtree.treeposition)
                    # if subtree.left_sibling == None:
                    #     left_most_child_position.append(subtree.treeposition)
                    


        ########################################### Missing node 
        copy_tree_2=tree.copy(deep=True) 
        pos = random.choice(tree_position[2:])
        #print(pos)
        for subtree in copy_tree_2.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
            if subtree.treeposition ==pos: 
                # print(subtree.label)
                subtree.prune()
        ###############################################################################
        
        ########################################### Superflous node with same label   
        copy_tree_5=tree.copy(deep=True) 
        pos2 = random.choice(tree_position[2:])
        for subtree in copy_tree_5.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0]="{}={}".format(subtree[0],sent[subtree[0]])
                    #sent[subtree[0]] = "{}={}".format(subtree[0],sent[subtree[0]])
            if subtree.treeposition ==pos2: 
                subtree.splicebelow(subtree.label)
        ###############################################################################
                
        ############################################# Wrong branch
        copy_tree_6=tree.copy(deep=True) 
        stop6 = False
        if len(right_most_child_position) == 1:
            right_most_pos = right_most_child_position[0]
        elif len(right_most_child_position) == 2:
            right_most_pos = right_most_child_position[1]
        else:
            right_most_pos = random.choice(right_most_child_position[2:])
        
        for subtree in copy_tree_6.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
                    #sent[subtree[0]] = "{}={}".format(subtree[0],sent[subtree[0]])
            if stop6 == False and subtree.treeposition ==right_most_pos:  #when the mothernode the rightmost
                for child in subtree:
                    if child.right_sibling == None and child.left_sibling != None: #if the current node is right most   
                        subtree.parent.append(child.detach()) #move this current node and its children to rightmost of oma node
                        stop6 = True
        #############################################################################################
        
        ############################################# Wrong label
        copy_tree_7=tree.copy(deep=True) 
        stop7 = False

        for subtree in copy_tree_7.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0]="{}={}".format(subtree[0],sent[subtree[0]])
                    #sent[subtree[0]] = "{}={}".format(subtree[0],sent[subtree[0]])
            if stop7 == False: #label case
                if subtree.label in ['NUC_N', 'AP', 'NUC_A']:
                    subtree.label = 'NUC'
                    stop7 = True
                if subtree.label in ['NUC']:
                    subtree.label = 'NUC_A'
                    stop7 = True
                if subtree.label in ['CORE_A', 'CORE_N']:
                    subtree.label = 'CORE'
                    stop7 = True
                if subtree.label in ['N']:
                    subtree.label = subtree.label+'-PROP'
                    stop7 = True
                if subtree.label in ['SENTENCE','NP','PP']:
                    subtree.label = subtree.label+'-PERI'
                    stop7 = True
                if subtree.label in ['V','OP']:
                    subtree.label = subtree.label+'-DEF'
                    stop7 = True
        #############################################################################################
 
            
        with open("input-test-sentences.bracketed", "a") as output: #parepare input for error test, top is correct and bottom is error tree
            output.write(str(tree)+'\n')
            output.write(str(copy_tree_2[0])+'\n')  #uncommen to create missing node error
            # output.write(str(copy_tree_5[0])+'\n')  #uncommen to create superflous label error
            # output.write(str(copy_tree_6[0])+'\n')  #uncommen to create wrong branch label error
            # output.write(str(copy_tree_7[0])+'\n')  #uncommen to create wrong label label error
            
with open("compare_output.txt", "w") as output: #prepare input
    output.close()
    


detail_approach.output_tree('input-test-sentences.bracketed') #run error detected/score

with open('output.bracketed') as f:
    reader = incrementaltreereader(f, strict=True, functions= 'add')
    diff_lists=[]
    for tree, sent, comment in reader:
        diff_list=[]
        for subtree in tree.subtrees():
            diff_list.append(subtree.label)
        diff_lists.append(diff_list)

######################################## Comparision, use only when compare 2 trees at a time used just ignore the "error out of range"
    with open("compare_output.txt", "a") as output: #compare
        for l in range(len(diff_lists)):
            if l % 2==0:
                output.write('-------------------- Sentence ' + str(int((l/2))+1)+ '---------------------------------------'+'\n') 
                
                c1 = Counter(diff_lists[l]) 
                c2 = Counter(diff_lists[l+1])
                diff1 = c1-c2
                diff2 = c2-c1
                output.write(str(list(diff1.elements()))+'\n')
                output.write(str(list(diff2.elements()))+'\n')
########################################################



    

        
            

            
    
    
                


