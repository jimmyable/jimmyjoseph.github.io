---
layout: post
title: "Find the single different char between two strings"
date: 2018-06-20 23:11:47
image: '/assets/img/'
description: 'Two strings walks into a bar, one of them has a special letter the other does not. We have to find which one'
tags:
- python
- find
- arrays
- strings
categories:
- Arrays and Strings
twitter_text:
---

# Find the single different char between two strings.
Again, simple puzzle. We just have to find the different unique letter the second string has and first one doesn't

## Tets cases
Let's define some test cases now.

- None input -> TypeError
- `'abcd'`, `'abcde'` -> `'e'`
- `'aaabbcdd'`, `'abdbacade'` -> `'e'`
## Method 1
{% highlight python %}
class Solution(object):

    def find_diff(self, s, t):
        if s and t is None:
            raise TypeError
        else:
            s1 = set(s)
            s2 = set(t)
            
            x=list(s2-s1)
            y=x.pop()
            
            return y
{% endhighlight %}

What I did was if s and t were None then we raise a *TypeError* as the testcase demanded.
Otherwise I use the `set()` function to find unique elements of both input strings. Take away the sets to find out what is left. The conversion to list and popping was just me improvising because for some reason `str()` didn't just convert a set into a string and I didn't research much further.

All in all what I made does infact pass the test cases.

## Method 2
{% highlight python %}
class Solution(object):

    def find_diff(self, str1, str2):
        if str1 is None or str2 is None:
            raise TypeError('str1 or str2 cannot be None')
        seen = {}
        for char in str1:
            if char in seen:
                seen[char] += 1
            else:
                seen[char] = 1
        print(seen)
        for char in str2:
            try:
                seen[char] -= 1
            except KeyError:
                return char
            if seen[char] < 0:
                return char
        
        return None

{% endhighlight %}

The **Correct** way to do it is shown here. Why this is more correct is that it works for even more abstract tests cases if there were any. For example if the input was "AAB", "AB" the left over string is "A" which this function can identify and my one won't. 

So how does it work?
Well, each charcter of String1 is looped and the letter along with its number of occurances are recorded.

In the second loop if the same letter is coming up then an occurance is taken off, if there is a *KeyError* this means its not in the `seen` dictionary and therefore unique so that can be spat out. If a letter does appear in the second string but higher occurances and therefore brings the `seen[char]`'s value to below 0, then it is the extra letter, it can be printed out.

## Unit Test

{% highlight python %}
# %load test_reverse_string.py
# %load test_str_diff.py
from nose.tools import assert_equal, assert_raises


class TestFindDiff(object):

    def test_find_diff(self):
        solution = Solution()
        assert_raises(TypeError, solution.find_diff, None, None)
        assert_equal(solution.find_diff('abcd', 'abcde'), 'e')
        assert_equal(solution.find_diff('aaabbcdd', 'abdbacade'), 'e')
        print('Success: test_find_diff')


def main():
    test = TestFindDiff()
    test.test_find_diff()


if __name__ == '__main__':
    main()

{% endhighlight %}














