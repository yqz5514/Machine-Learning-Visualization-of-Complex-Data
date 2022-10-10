#%%
# library
import pandas as pd
import re

#%%
with open('1976.txt', 'r') as f:
    text = f.read()

#%%
text[0:5]
# %%
pattern_WKU = re.compile(r'WKU\s\s(RE)?\d+')
matches = pattern_WKU.finditer(text)
list = []
for match in matches:
    value = match.group()

    list.append(value)
print(list)

#%%
len(list) #1379
#%%
abs_match = list
# %%
type(abs_match)

#%%
abs_match

#%%
print(list)
## Looping over regex matches inside text
#for match_num, match in enumerate(re.finditer(reg, test, re.MULTILINE), start=1):
#    print(f"Match {match_num} was found at {match.start()}-{match.end()}: {match.group()}")
# %%
sum(1 for _ in re.finditer(pattern_WKU,text))
# 1379 products
# %%
type(match.group())
#%%
pattern_ISD = re.compile(r'ISD\s\s(1976)\d+')
matches = pattern_ISD.finditer(text)
for match in matches:
    print(match.group()) # same value for 19760106_wk1
#%%
ISD = []
ISD.append(match.group())
#%%
sum(1 for _ in re.finditer(pattern_ISD,text))
#8363
#becasue there are many ISD value besides the one associate with publish time
#publish time start with 1976 #1379
#%%%
#pattern_ABS = re.compile(r'ABST\s.+\s.+[A-Z]$')
#matches = pattern_ABS.finditer(text)
#for match in matches:
 #   print(match.group())


#%%
pattern_ABS = re.compile(r'ABST\n*\n*((?:\n.*)+?)(?=\n[A-Z]{4}|\Z)')
matches = pattern_ABS.finditer(text)
#for match in matches:
#    print(match.group())     
#    .*\n*: Match rest of the line followed by 0 or more line breaks
#((?:\n.*)+?): Capture group 1 to capture our text which 1 or lines of everything until next condition is satisfied
#(?=\nAREA:|\Z): Assert that we have a line break followed by AREA: or end of input right ahead of the current position

#%%
ABST = []
for match in matches:
    val = match.group()
    ABST.append(val)
#print(ABST)
#%%
#abst = pd.DataFrame({'ABST':ABST})
#abst.to_excel('abst_check.xlsx')
#%%
sum(1 for _ in re.finditer(pattern_ABS,text))
# Search for 'ABST' 1379
# search in text file by case is 1379
# search by paragraph with 1384 
# add'\n' after ABST match #1379
#%%

#%%
#test
with open('test.txt', 'r') as f:
    text1 = f.read()

#%%
pattern_ABS = re.compile(r'par')
matches = pattern_ABS.finditer(text1)
#for match in matches:
#    print(match.group())     
#%%
sum(1 for _ in re.finditer(pattern_ISD,text))

#%%
#ICL
pattern_ICL = re.compile(r'ICL\s\s\w+')
matches = pattern_ICL.finditer(text)
#for match in matches:
#    print(match.group())
#%%
sum(1 for _ in re.finditer(pattern_ICL,text))
# search ICL\s\s\w+ #2023
# some aplication has multiple line of ICL info

    
# %%
result_1976 = pd.DataFrame({'Pattent ID':abs_match, 'Publish Date':ISD, 'Abstract':ABST})
result_1976.to_excel('demo.xlsx')
#%%
print(result_1976.head())
# %%
# publish day is same in one file?
# CLAS problems