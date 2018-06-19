---
layout: post
title: "Permutation of Characters"
date: 2018-03-07 07:17:47
image: '/assets/img/'
description: 'Checking if one string is a permutation of another'
tags:
- python
- permutation
- test
- arrays
- strings
categories:
- Problems
- Arrays and Strings
twitter_text:
---

# Permuatation of characters
So what are we doing here? we are checking if both strings contains the same characters but in a different order i.e permutation.

## Test cases
The following test cases must produce these results.
- One or more None inputs *returns* **False**
- One or more empty strings *returns* **False**
- `'Nib'`, `'bin'` *returns* **False**
- `'act'`, `'cat'` *returns* **True**
- `'a ct'`, `'ca t'` *returns* **True**

## Code
{% highlight python %}
class Permutations(object):

    def is_permutation(self, str1, str2):
        if str1 is None or str2 is None:
            return False
        return sorted(str1) == sorted(str2)
{% endhighlight %}

The function above is taking in the two strings we can send in. The first check makes sure neither of the strings are `None`, to satisfy the first two test cases.

Next we use a standard Python function [sorted](https://docs.python.org/3/howto/sorting.html). This function sorts out the input, in an ascending order by default. You can sort by descending order by specifying additional parameters to the fucntion such as `reverese=True`. If you feel brave you can even use `keys` and `lambdas` to create very advanced sorting functions.

So, the sorted fucntion sorts out the input string. The final line must give us a correct answer to the rest of our test cases now. Note that this algorithm is case sensitive, you can always convert both strings to a specified case before the equality comparison to change this.

## Unit Test
{% highlight python %}
from nose.tools import assert_equal


class TestPermutation(object):

    def test_permutation(self, func):
        assert_equal(func(None, 'foo'), False)
        assert_equal(func('', 'foo'), False)
        assert_equal(func('Nib', 'bin'), False)
        assert_equal(func('act', 'cat'), True)
        assert_equal(func('a ct', 'ca t'), True)
        print('Success: test_permutation')


def main():
    test = TestPermutation()
    permutations = Permutations()
    test.test_permutation(permutations.is_permutation)
    try:
        permutations_alt = PermutationsAlt()
        test.test_permutation(permutations_alt.is_permutation)
    except NameError:
        # Alternate solutions are only defined
        # in the solutions file
        pass


if __name__ == '__main__':
    main()
{% endhighlight %}
