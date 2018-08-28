---
layout: post
title: "Reverse characters in a string(in-place)"
date: 2018-06-19 23:02:47
image: '/assets/img/'
description: 'Implement a function to reverse a string (a list of characters), in-place.'
tags:
- python
- reverse
- arrays
- strings
categories:
- Arrays and Strings
twitter_text:
---

# Reverse characters in a string(in-place)
Self explanatory really, pretty much if x=["x","y","Z"] was fed in, then the output should be x=["z","y","x"]. 


## Tets cases
Let's define some test cases now.

- None -> None
- [''] -> ['']
- ['f', 'o', 'o', ' ', 'b', 'a', 'r'] -> ['r', 'a', 'b', ' ', 'o', 'o', 'f']

## Method 1
{% highlight python %}
class ReverseString(object):

    def reverse(self, chars):
        #print(chars, "+++")
        rev=[]
        if chars is None:
            return chars
        else:
            for i in range (len(chars)):
                rev.append(chars.pop())
            #print(rev,"___")
            return rev
{% endhighlight %}

This method is pretty easy, I made a second variable and for the length of the incoming string I pop out the last element and add it to the new list. Because Popping happens in a LIFO this method works. But this is not "in-place".

## Method 2
{% highlight python %}
from __future__ import division


class ReverseString(object):

    def reverse(self, chars):
        if chars:
            size = len(chars)
            for i in range(size // 2):
                print(i,"++++")
                print(chars[i],chars[size - 1 - i],chars[size - 1 - i],chars[i])
                chars[i], chars[size - 1 - i] = chars[size - 1 - i], chars[i]
        return chars

{% endhighlight %}

"In-place" means, reversing the string without making using any temporary storage. This algorithm is more efficinet. The print statements show how the rearranging is carried out. 

Essentially, The first and last charatcers are switched, then the second and second last, then third and third last, and so on until the middle. At this point the whole string is flipped and we have achieved the goal.

## Unit Test

{% highlight python %}
# %load test_reverse_string.py
from nose.tools import assert_equal


class TestReverse(object):

    def test_reverse(self, func):
        assert_equal(func(None), None)
        assert_equal(func(['']), [''])
        assert_equal(func(
            ['f', 'o', 'o', ' ', 'b', 'a', 'r']),
            ['r', 'a', 'b', ' ', 'o', 'o', 'f'])
        print('Success: test_reverse')

    def test_reverse_inplace(self, func):
        target_list = ['f', 'o', 'o', ' ', 'b', 'a', 'r']
        func(target_list)
        assert_equal(target_list, ['r', 'a', 'b', ' ', 'o', 'o', 'f'])
        print('Success: test_reverse_inplace')


def main():
    test = TestReverse()
    reverse_string = ReverseString()
    test.test_reverse(reverse_string.reverse)
    test.test_reverse_inplace(reverse_string.reverse)


if __name__ == '__main__':
    main()
{% endhighlight %}














