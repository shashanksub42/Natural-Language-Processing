
# coding: utf-8

# In[ ]:

import nltk
import re
import sys
import numpy as np
from astropy.table import Table, Column
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[ ]:

input_src = input("Enter source file: ")
input_tgt = input("Enter targte file: ")


# In[ ]:

src = open(input_src)
tgt = open(input_tgt)


# In[ ]:

srcRaw = src.read()
tgtRaw = tgt.read()


# In[ ]:

new_src = []
new_tgt = []
Source = []
Target = []


# In[ ]:

def tokenize_src(srcList):
    srcList = srcRaw.lower().split()
    for word in srcList:
        if len(word) == 1:
            new_src.append(word)
        elif not word[0].isalnum() and not word[len(word)-1].isalnum(): #Checks for first and last characters
            #Checks whether token has consequtive characters are non alnum
            count = 0
            k = 0
            for k in word:
                if not k.isalnum():
                    count += 1
                if count == len(word):
                    new_tgt.append(word)
                    continue
            i = 0
            while not word[i].isalnum(): #Checks start of string
                new_src.append(word[i])
                i += 1
            word = word[::-1] #Reverse string
            j = 0
            while not word[j].isalnum(): #Checks at the end of string
                j += 1 
            word = word[::-1]
            new_src.append(word[i:len(word)-j]) #Append alnum to new list
            for x in word[len(word)-j:len(word)]:
                new_src.append(x) #Append non alnum to new list
        elif not word[0].isalnum(): #Checks only for first character as non alnum
            i = 0
            while not word[i].isalnum():
                new_src.append(word[i]) #Append to new list
                i += 1        
        elif not word[len(word)-1].isalnum(): #Checks only for last character as non alnum
            word = word[::-1]
            j = 0
            while not word[j].isalnum():
                j += 1
            word = word[::-1]
            new_src.append(word[0:len(word)-j]) #Append to new list
            for x in word[len(word)-j:len(word)]:
                new_src.append(x)
        else:
            new_src.append(word) #If there are no non alnum characters in string, append word as is
    
    j = 0
    preword = ""
    postword = ""
    #Check for apostrophe in the new list
    #After checking, append all tokens to another list - 'Source'
    for word in new_src:
        if word[len(word)-2] == "\'" and word[len(word)-1] == "t":
            preword = word[j:len(word)-3]
            postword = "not"
            Source.append(preword)
            Source.append(postword)
        elif word[len(word)-2] == "\'" and word[len(word)-1] == "s":
            preword = word[j:len(word)-2]
            postword = "\'s"
            Source.append(preword)
            Source.append(postword)
        elif word[len(word)-2] == "\'" and word[len(word)-1] == "m":
            preword = word[j:len(word)-2]
            postword = "am"
            Source.append(preword)
            Source.append(postword)
        else:
            Source.append(word)
    
    return Source


# In[ ]:

def tokenize_tgt(tgtList):
    tgtList = tgtRaw.lower().split()
    for word in tgtList:
        if len(word) == 1:
            new_tgt.append(word)
        elif not word[0].isalnum() and not word[len(word)-1].isalnum(): #Checks for first and last characters
            #Checks whether token has consequtive characters are non alnum
            k = 0
            count = 0
            for k in word:
                if not k.isalnum():
                    count += 1
            if count == len(word):
                new_tgt.append(word)
                continue
            i = 0
            while not word[i].isalnum(): #Checks start of string
                new_tgt.append(word[i])
                i += 1
            word = word[::-1]              #Reverse string
            j = 0
            while not word[j].isalnum():    #Checks at the end of string
                j += 1 
            word = word[::-1]
            new_tgt.append(word[i:len(word)-j])   #Append alnum to new list
            for x in word[len(word)-j:len(word)]: #Checks last part of string
                new_tgt.append(x)                  #Append to non alnum to new list
        elif not word[0].isalnum(): #Checks only first character
            i = 0
            while not word[i].isalnum():
                new_tgt.append(word[i])    #Append start non alnum chars to new list
                i += 1        
        elif not word[len(word)-1].isalnum(): #Checks only last character
            word = word[::-1]
            j = 0
            while not word[j].isalnum():
                j += 1
            word = word[::-1]
            new_tgt.append(word[0:len(word)-j]) #Append last non alnum chars to new list
            for x in word[len(word)-j:len(word)]:
                new_tgt.append(x)
        else:
            new_tgt.append(word)               #If none of the characters in string are non alnum, append word as is
    j = 0
    preword = ""
    postword = ""
    for word in new_tgt:
        if word[len(word)-2] == "\'" and word[len(word)-1] == "t":
            preword = word[j:len(word)-3]
            postword = "not"
            Target.append(preword)
            Target.append(postword)
        elif word[len(word)-2] == "\'" and word[len(word)-1] == "s":
            preword = word[j:len(word)-2]
            postword = "\'s"
            Target.append(preword)
            Target.append(postword)
        elif word[len(word)-2] == "\'" and word[len(word)-1] == "m":
            preword = word[j:len(word)-2]
            postword = "am"
            Target.append(preword)
            Target.append(postword)
        else:
            Target.append(word)
    return Target


# In[ ]:

def create_edit_table(rows, cols, maxima_count):
    score_matrix = np.zeros((cols, rows))
    # Fill the scoring matrix.
    max_score = 0
    max_pos   = None    # The row and columbn of the highest score in matrix.
    for i in range(1, score_matrix.shape[0]):
        for j in range(1, score_matrix.shape[1]):
            score = calc_score(score_matrix, i, j)
            if score > max_score:
                max_score = score
                max_pos   = (i, j)
#             print(max_score)
            score_matrix[i][j] = score
    row_len = len(score_matrix)
    col_len = len(score_matrix[0])
#     print("Maxima:", max_pos)
   
    print("Max Score:", max_score)
    i = 0
    j = 0
    for i in range(row_len):
        for j in range(col_len):
            if score_matrix[i][j] == max_score:
                print("Maxima: (", i, ",",j,")")
                maxima_count.append((i,j))
            
#     assert max_pos is not None, 'the x, y position with the highest score was not found'

    return score_matrix, max_pos, maxima_count


# In[ ]:

def calc_score(matrix, x, y):
    similarity = match if Source[x - 1] == Target[y - 1] else mismatch

    diag_score = matrix[x - 1, y - 1] + similarity
    up_score   = matrix[x - 1, y] + gap
    left_score = matrix[x, y - 1] + gap
    max_score = max(0, diag_score, up_score, left_score)
    if diag_score == max_score and diag_score != 0:
        backtrace[x][y] = 2                             #Diagonal backtrace
    elif up_score == max_score and up_score != 0:
        backtrace[x][y] = 1                             #Upward backtrace
    elif  left_score == max_score and left_score != 0:
        backtrace[x][y] = 3  
                                          
    return max(0, diag_score, up_score, left_score)


# In[ ]:

def trace_back(src_str,tgt_str,loc_X,loc_Y,backtrace_matrix,x,y):
   
    if backtrace_matrix[x,y]=='DI' :
        src_str.append(Source[x - 1])
        tgt_str.append(Target[y - 1])
        x -= 1
        y -= 1
        loc_X.append(x)
        loc_Y.append(y)
        trace_back(src_str, tgt_str,loc_X,loc_Y, backtrace_matrix, x, y)
    elif backtrace_matrix[x,y]=='UP':
        src_str.append(Source[x - 1])
        tgt_str.append('-')
        x -= 1
        loc_X.append(x)
        loc_Y.append(y)
        trace_back(src_str, tgt_str,loc_X,loc_Y, backtrace_matrix, x, y)
    elif backtrace_matrix[x,y]=='LT':
        src_str.append('-')
        tgt_str.append(Target[y - 1])
        y -= 1
        loc_X.append(x)
        loc_Y.append(y)
        trace_back(src_str, tgt_str,loc_X,loc_Y, backtrace_matrix, x, y )
    return src_str,tgt_str,loc_X,loc_Y


# In[ ]:

def alignment(max_pos, backtracematrix, max_count):
    i=0
     
    for k in max_count:
        x,y = k
        a = []
        b = []
        c = []
        d = []
        p,q,r,s= trace_back(a, b,c,d, backtracematrix, int(x), int(y))
        p=p[::-1]
        q=q[::-1]
        print("    Alignment ",i,"( length ",len(p),") :\n")
        print("       Source at ",r[len(r)-1]," ",p)
        print("       Target at ",s[len(s)-1]," ",q)
        print("     Edit Action     ",Edit_Action(p,q))
        print("\n\n")
        i += 1


# In[ ]:

def Edit_Action(source,target):
    editAction=[]
    for pos,i in enumerate(source):
        if i==target[pos]:
            editAction.append(" ")
        elif i=="-" and target[pos].isalnum():
            editAction.append(" i ")
        elif i.isalnum() and target[pos]=="-":
            editAction.append(" d ")
        else:
            editAction.append(" s ")
    return editAction


# In[ ]:

if __name__ == '__main__':
#     file1 = input("Enter source file :")
#     file2= input("Enter target file :")

    src_raw_tkn=srcRaw                  #Raw source tokens
    tgt_raw_tkn=tgtRaw                  #Raw target tokens
    Source = tokenize_src(src_raw_tkn)  #Normalized source tokens
    seq1 = ' '.join(Source)
    Target = tokenize_tgt(tgt_raw_tkn)  #Normalized target tokens
    seq2 = ' '.join(Target)

    match = 2
    mismatch = -1
    gap = -1
    
    maxima_count = []                   #Maintains a list of max positions of max score
    rows = len(seq2.split()) + 1
    cols = len(seq1.split()) + 1
    split_src = seq1.split()
    split_tgt = seq2.split()
    backtrace = np.zeros((cols, rows))
    
   
    print("University of Central Florida\n")
    print("CAP6640 Spring 2018 - Dr. Glinos\n\n")
    print("Text Similarity Analysis by Shashank Subramanian\n")

    print("Source file: gene-src.txt\n")
    print("Target file: gene-tgt.txt\n")

    print("Raw Tokens:\n")
    print("Source:", (srcRaw), "\n")
    print("Target:", (tgtRaw), "\n")

    print("Normalized Tokens:\n")
    print("Source:", Source, "\n")
    print("Target:", Target, "\n")
    
    Source.insert(0, "#")
    Target.insert(0, "#")
    NewSrc = [Source[i][0:3] for i in range(0, len(Source))]
    NewTgt = [Target[i][0:3] for i in range(0, len(Target))]

    src_lst = [s for s in range(0, len(NewSrc))]
    tgt_lst = [t for t in range(0, len(NewTgt))]

    
    row_list = []
    row_list.append(src_lst)
    row_list.append(NewSrc)


    col_list = []
    col_list.append(tgt_lst)
    col_list.append(NewTgt)
    print("\nEdit Distance Table:\n")
    edit, max_pos,max_count = create_edit_table(rows, cols, maxima_count)
    pd.options.display.float_format = '{:,.0f}'.format
    final_edit_table = pd.DataFrame(edit, index=row_list, columns=col_list)
    print(final_edit_table)
    print('\n\n')
    
    print("Backtrace Table:\n\n")
    newbacktrace = np.empty((cols, rows)).astype(object)
    back=backtrace.astype(int)

    for i in range(0, back.shape[0]):                    #Adds DI, UP, LT corresponding to old backtrace table to new backtrace table
            for j in range(0, back.shape[1]):
                if back[i,j]==1:
                    newbacktrace[i, j] ="UP"
                elif back[i,j]==3:
                    newbacktrace[i, j] = "LT"
                elif back[i,j]==2:
                    newbacktrace[i, j] = "DI"
                else:newbacktrace[i,j]=""


    final_back_table = pd.DataFrame(newbacktrace, index=row_list, columns=col_list)

    print(final_back_table)
    print("\n\n")
    print("Maximal-similarity alignments: \n\n")
    alignment(max_pos, newbacktrace, max_count)


# In[ ]:



