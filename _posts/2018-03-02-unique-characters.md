---
layout: post
title: "Unique Characters"
date: 2018-03-02 20:40:47
image: '/assets/img/'
description: 'Creating an algorithm that can decide if a given string has all unique characters'
tags:
- python
- unique
- test
- arrays
- strings
categories:
- Problems
- Arrays and Strings
twitter_text:
---

# Unique Characters
We want to use python to build an algorithm that can check if a given string contains all unique characters. Run the code with the unit test to confirm its validity.

## Test cases
We will use the following test cases to determine if the algorithm is correct.
- `None` *returns* **False**
- `''` *returns* **True**
- `'foo'` *returns* **False**
- `'bar'` *returns* **True**

## Code
{% highlight python %}
class UniqueChars(object):

    def has_unique_chars(self, string):
        if string == None:
            return False
        return len(set(string))==len(string)
{% endhighlight %}
Okay so whats happening here? It's actually a very simple answer to a relatively simple problem. First the equation takes in `string`. This is the word that is fed into the function, you can see how, in the unit test file.

If the `string ==  None` then we can just return **False**, because nothing was really fed in.

Before explaining the next step. In Python a [set](https://docs.python.org/3/tutorial/datastructures.html#sets)(sometimes called unordered set) is a data type that can only contain *unique* elements. This means if I made a set of the word **apple** using `set('apple')` then printing it back out gives `{'e', 'p', 'a', 'l'}`, the individual characters. P.S you can store unique words by issuing `set(['apple'])`, in which case the unique element is the whole word.

Now that we have that straight, the last line `len(set(string))==len(string)` checks if the length of the unique characters are the same as the length of the original string. If they are equal then the original string contained only unique characters, which is returned.

## Unit Test
{% highlight python %}
from nose.tools import assert_equal


class TestUniqueChars(object):

    def test_unique_chars(self, func):
        assert_equal(func(None), False)
        assert_equal(func(''), True)
        assert_equal(func('foo'), False)
        assert_equal(func('bar'), True)
        print('Success: test_unique_chars')


def main():
    test = TestUniqueChars()
    unique_chars = UniqueChars()
    test.test_unique_chars(unique_chars.has_unique_chars)
    try:
        unique_chars_set = UniqueCharsSet()
        test.test_unique_chars(unique_chars_set.has_unique_chars)
        unique_chars_in_place = UniqueCharsInPlace()
        test.test_unique_chars(unique_chars_in_place.has_unique_chars)
    except NameError:
        # Alternate solutions are only defined
        # in the solutions file
        pass


if __name__ == '__main__':
    main()
{% endhighlight %}







