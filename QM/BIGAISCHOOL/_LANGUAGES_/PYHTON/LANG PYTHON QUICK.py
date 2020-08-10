#!/usr/bin/env python
# coding: utf-8

# # Python QUICK
# 

# ## Integers

# In[1]:


a = 66
type(a)


# In[2]:


b = 1000
b.bit_length()


# In[4]:


a+b #This is commentary.Result of a+b should be 1066


# ## Floats

# In[5]:


c= 1.6
d=1.
type(c)
type(d)


# ## Boolean

# In[6]:


6>4


# In[8]:


type(False)


# In[9]:


not True


# In[3]:


not False


# In[11]:


(3>4) and (2<1)


# In[15]:


int(True)


# In[16]:


int(False)


# ## Strings

# In[11]:


s = " string this is it "


# In[8]:


s.capitalize()


# In[20]:


s.split()


# In[9]:


s


# In[21]:


s.find("this")


# In[12]:


print(s+s)


# In[4]:


"what is happening {:d} times?".format(12)


# In[29]:


"what is happening {:04d} times?".format(12)


# ## Date Time

# In[4]:


from datetime import datetime
dt = datetime.strptime('04/22/2017 15:00:00'.replace("'", ""),'%m/%d/%Y %H:%M:%S')
print(dt)
type(dt)


# ## If , While

# In[12]:


if 10<20:
    print("Yes,condition fulfiled")


# In[14]:


i =0
while i <4:
    print("still in the while loop")
    i = i + 1 # or i += 1
print("outsite of loop")


# In[1]:


range(1,6)


# ## for loop
# 

# In[2]:


for x in range(1, 5):
    print(x+1)


# In[26]:


l = [i+i for i in range(8)]
print(l)
type(l)


# ## Function def

# In[28]:


def Add(x,y):
    return x+y


# In[9]:


def Kwadratowa(x):
    return x*x


# In[10]:


Kwadratowa(9)


# ## Printing Numbers and using function

# In[18]:


summa = 0
for x in range(1, 5):
    summa = Add(summa,x)
    print(f"Adding {x}, summa so far is {summa}")


# ## Tuple

# In[7]:


tup = (4, 2.4, 'python')
type(tup)


# In[8]:


tup[2]


# In[10]:


tup.count(2.4)


# In[13]:


tup.index('python')


# ## Lists
# 

# In[17]:


li = list(tup)
print(li)
type(li)


# In[19]:


a = ['a', 'b', 'c', 'd']
print(a)


# In[20]:


for x in a:
    print(x)


# In[21]:


for x,y in enumerate(a):
    print(f"{x}:{y}")


# In[21]:


# Manually add items, lists allow duplicates
list1 = []
list1.append('a')
list1.append('b')
list1.append('c')
list1.append('c')
list1.extend([4.6, 6.6])
list1.insert(3,'uhu')
print(list1)
print(list1[2:4])


# ## Adding , Removing items from list

# In[24]:


# Insert
c = ['a', 'b', 'c']
c.insert(0, 'blablabla')
print(c)
# Remove
c.remove('b')
print(c)
# Remove at index
del c[0]
print(c)


# ## Maps/Dictionaries/Sets/Hash Tables and Functional Programming

# In[40]:


d = {'name': "Alex", 'address':"Street magic"}
print(d)
print(d['name'])

if 'name' in d:
    print("Name is defined and the name is " + d['name'])

if 'age' in d:
    print("age defined")
else:
    print("age undefined")


# In[41]:


d = {'name': "Alex", 'address':"Street magic"}
# All of the keys
print(f"Key: {d.keys()}")

# All of the values
print(f"Values: {d.values()}")


# In[42]:


for item in d.
    print(item)


# In[23]:


# Manually add items, sets do not allow duplicates
# Sets add, lists append.
set1 = set()
set1.add('a')
set1.add('b')
set1.add('c')
set1.add('c')
print(set1)


# In[44]:


li = [4, 2, 10, 2, 1, 10, 0, 6, 0, 8, 10, 9, 2, 4, 7, 8, 10, 8, 8, 2]
type(li)


# In[47]:


s = set(li)# Remove duplicates
print(s)


# In[33]:


list(map(Add, range(10),range(10)))


# In[36]:


list(map(lambda x,y: x+y, range(10), range(10)))

