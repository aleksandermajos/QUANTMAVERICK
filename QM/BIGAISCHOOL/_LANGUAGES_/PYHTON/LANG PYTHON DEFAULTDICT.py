#!/usr/bin/env python
# coding: utf-8

# In[2]:


from collections import defaultdict


# In[3]:


def_dict = defaultdict(list)  # Pass list to .default_factory
def_dict['one'] = 1  # Add a key-value pair
def_dict['missing']  # Access a missing key returns an empty list

def_dict['another_missing'].append(4)  # Modify a missing key
def_dict


# In[4]:


dd = defaultdict(list)
dd['key'].append(1)

dd['key'].append(2)

dd['key'].append(3)
dd


# In[5]:


std_dict = dict(numbers=[1, 2, 3], letters=['a', 'b', 'c'])
std_dict

def_dict = defaultdict(list, numbers=[1, 2, 3], letters=['a', 'b', 'c'])
def_dict

std_dict == def_dict


# In[ ]:




