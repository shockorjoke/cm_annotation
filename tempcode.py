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


from torch import NoneType

file = open('partage_de.bracketed')
  
# read the content of the file opened
content = file.readlines()
  
# read 10th line from the file
with open("test-sentences.bracketed", "w") as output: # output tree on output.bracketed
    output.write(content[1])
    output.write(content[11])
    output.write(content[21])
    output.write(content[31])
    output.write(content[41])
    output.write(content[51])
    output.write(content[61])
    output.write(content[71])
    output.write(content[81])
    output.write(content[91])
    
with open("input-test-sentences.bracketed", "w") as output: #prepare input
    output.close()

with open('test-sentences.bracketed') as f: #edit sentence for input
    reader = incrementaltreereader(f, strict=True, functions= 'add')
    for tree, sent, comment in reader:
        for subtree in tree.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
                else:
                    continue
        
        copy_tree_1=tree.copy(deep=True) #missing a node
        for subtree in copy_tree_1.subtrees(): 
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
                else:
                    if subtree.label == 'CLAUSE' and subtree.parent.label == 'SENTENCE':
                        subtree.prune()
                        
        copy_tree_2=tree.copy(deep=True) #superfluous node with same label 
        for subtree in copy_tree_2.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
                else:
                    if subtree.label == 'SENTENCE' and subtree.parent == None:
                        subtree.splicebelow('CLAUSE')
                        
        copy_tree_3=tree.copy(deep=True) #superfluous node with different label 
        for subtree in copy_tree_3.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
                else:
                    if subtree.label == 'SENTENCE' and subtree.parent == None:
                        subtree.splicebelow('CORE')
                        
        copy_tree_4=tree.copy(deep=True) #Wrong label
        for subtree in copy_tree_4.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
                else:
                    if subtree.label == 'CORE' and subtree.parent.label == 'CLAUSE' :
                        subtree.label = 'CORE_N'
                        
        copy_tree_5=tree.copy(deep=True) #wrong branch TODO
        for subtree in copy_tree_5.subtrees():
            for child in subtree:
                if isinstance(child,int):
                    subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
            if subtree.label == 'NP' and subtree.parent.label == 'CORE':
                subtree.parent.parent.append(subtree.detach())



        # copy_tree_6=tree.copy(deep=True)      
        # for subtree in copy_tree_6.subtrees():
        #     for child in subtree:
        #         if isinstance(child,int):
        #             subtree[0] = "{}={}".format(subtree[0],sent[subtree[0]])
        #         else:
        #             if subtree.label == 'SENTENCE' and subtree.parent == None:
        #                 subtree.splicebelow('CLAUSE')
        with open("input-test-sentences.bracketed", "a") as output:
            output.write(str(tree)+'\n')
            output.write(str(copy_tree_1)+'\n')
            output.write(str(tree)+'\n')
            output.write(str(copy_tree_2)+'\n')
            output.write(str(tree)+'\n')
            output.write(str(copy_tree_3)+'\n')
            output.write(str(tree)+'\n')
            output.write(str(copy_tree_4)+'\n')
            output.write(str(tree)+'\n')
            output.write(str(copy_tree_5)+'\n')
            # output.write(str(tree)+'\n')
            # output.write(str(copy_tree_6)+'\n')
            
with open("compare_output.txt", "w") as output: #prepare input
    output.close()

detail_approach.output_tree('input-test-sentences.bracketed') #run anotation

with open('output.bracketed') as f:
    reader = incrementaltreereader(f, strict=True, functions= 'add')
    diff_lists=[]
    for tree, sent, comment in reader:
        diff_list=[]
        for subtree in tree.subtrees():
            diff_list.append(subtree.label)
        diff_lists.append(diff_list)
    # print(diff_lists[2])
    # print(diff_lists[3])
    with open("compare_output.txt", "a") as output: #compare
        for l in range(len(diff_lists)):
            if l % 2==0:
                output.write('-------------------- Sentence ' + str(int((l/10))+1)+ '---------------------------------------'+'\n')
                output.write(str(list(sorted(set(diff_lists[l])-set(diff_lists[l+1]),reverse=True)))+'\n')
                output.write(str(list(sorted(set(diff_lists[l+1])-set(diff_lists[l]),reverse=True)))+'\n')
                # output.write(str(list(set(diff_lists[l])-set(diff_lists[l+1])))+'\n')
                # output.write(str(list(set(diff_lists[l+1])-set(diff_lists[l])))+'\n')   

with open('top_and_bottom_ten.txt', 'w') as tb: 
    tb.close()
    
# with open('top_and_bottom_ten.txt', 'w') as tb:
    
    
    # self_rules_posibility,left_neighbour_rules_posibility,right_neighbour_rules_posibility,_,_ = detail_approach.confident_measure()
    # df_self=pd.DataFrame.from_dict(self_rules_posibility, orient='index', columns=['score'])
    # tb.write('---Self_rule---'+'\n')
    # tb.write(str(df_self.score.nlargest(10))+'\n')
    
    # df_left=pd.DataFrame.from_dict(left_neighbour_rules_posibility, orient='index', columns=['score'])
    # tb.write('---Left_rule---'+'\n')
    # tb.write(str(df_left.score.nlargest(10))+'\n')
    
    # df_right=pd.DataFrame.from_dict(right_neighbour_rules_posibility, orient='index', columns=['score'])
    # tb.write('---Right_rule---''\n')
    # tb.write(str(df_right.score.nlargest(10))+'\n')
    
    # tb.write('---Self_rule---'+'\n')
    # tb.write(str(df_self.score.nsmallest(10))+'\n')
    
    # df_left=pd.DataFrame.from_dict(left_neighbour_rules_posibility, orient='index', columns=['score'])
    # tb.write('---Left_rule---'+'\n')
    # tb.write(str(df_left.score.nsmallest(10))+'\n')
    
    # df_right=pd.DataFrame.from_dict(right_neighbour_rules_posibility, orient='index', columns=['score'])
    # tb.write('---Right_rule---''\n')
    # tb.write(str(df_right.score.nsmallest(10))+'\n')

    

        
            

            
    
    
                


