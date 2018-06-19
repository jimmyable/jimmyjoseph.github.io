---
layout: post
title: "Rotational Characters"
date: 2018-03-12 07:41:47
image: '/assets/img/'
description: 'Checking if one string is a rotation of another'
tags:
- python
- rotational
- test
- arrays
- strings
categories:
- Problems
- Arrays and Strings
twitter_text:
---

# Rotational Characters
If a word is rotated then it keeps its oringal order but the sequence might start or end at a different point. For example, `OGOGOG` is a rotation of `GOGOGO`, but not `GGGOOO` or `GGOOGO`. Another example is `Tree` which can be rotated to get `reeT`. 

## Tets cases
Let's define some test cases now.

- Any strings that differ in size *returns* **False**
- `None`, `'foo'` *returns* **False** (any None results in False)
- `' '`, `'foo'` *returns* **False**
- `' '`, `' '` *returns* **True**
- `'foobarbaz'`, `'barbazfoo'` *returns* **True**

## Code
{% highlight python %}
class Rotation(object):

    def is_substring(self, s1, s2):
        return s1 in s2

    def is_rotation(self, s1, s2):
        if s1 is None or s2 is None:
            return False
        if len(s1) != len(s2):
            return False
        return self.is_substring(s1, s2 + s2)
{% endhighlight %}

What do we have here? We are only going to call the `is_rotation` function directly. It accepts two strings. To satisfy the first test case, we check if any of the strings are `None`, if so automatically return `False`. 

Next the `len()` function is used to check if both strings are the same size. If they are not the same size they cannot be permutations of eachother for obvious reasons.

Next we do a clever trick. We send the first string and a conjuntcion of second string repeated as the second argument, to `is_substring`. Why?

Looking at earlier example of `Tree` and `reeT`. If we add both stirngs we get `reeTreeT`, observe how by adding string2 to iteself we see string1 inside itself? This can only happen is both strings are rotational equaivalents of eachother. 

Looking back at the code, the `is_substring` checks exactly this, it looks for the first string in the repeated second string. If a match is found then the function returns `True`.

## Unit Test

{% highlight python %}
from nose.tools import assert_equal

class TestRotation(object):

    def test_rotation(self):
        rotation = Rotation()
        assert_equal(rotation.is_rotation('o', 'oo'), False)
        assert_equal(rotation.is_rotation(None, 'foo'), False)
        assert_equal(rotation.is_rotation('', 'foo'), False)
        assert_equal(rotation.is_rotation('', ''), True)
        assert_equal(rotation.is_rotation('foobarbaz', 'barbazfoo'), True)
        print('Success: test_rotation')


def main():
    test = TestRotation()
    test.test_rotation()


if __name__ == '__main__':
    main()
{% endhighlight %}














