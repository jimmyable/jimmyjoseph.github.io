---
layout: post
title: "Given an array, find the two indices that sum to a specific value."
date: 2018-06-22 10:24:47
image: '/assets/img/'
description: 'Find two numbers in a given list, check if it sums to tha target value, if so print out the indices of those two numbers'
tags:
- python
- find
- arrays
- strings
categories:
- Problems
- Arrays and Strings
twitter_text:
---

# Given an array, find the two indices that sum to a specific value.
Try to understand Method #2, its pretty clever imo.

## Tets cases
Let's define some test cases now.

- None input -> TypeError
- `[]` -> ValueError
- `[1, 3, 2, -7, 5], 7 ` -> `[2, 4]`

## Method 1
{% highlight python %}
import itertools
class Solution(object):

    def two_sum(self, nums, val):
        if nums is None and val is None:
            raise TypeError
        if nums ==[] and val ==0:
            raise ValueError
            
        else:
            coffee = []
            x = list(itertools.combinations_with_replacement(nums,2))
            #print(x)
            for i in range (len(x)):
                y=x.pop()
                z=y[0]+y[1]
                #print(z)
                if z == val:
                    coffee.append((y[0],y[1]))
            coffee = list(coffee.pop())
            print(coffee)
            
            answer =[]
            for j in range (len(coffee)):
                answer.append(nums.index(coffee[j]))
                print(answer)
            return answer
{% endhighlight %}

My naive method, first I do the usual checks if input is None or empty and raise the appropirate errors. 

Then I create(initilaise) a list called `coffee` which I will use to store which numbers make up the target.
I also imported `itertools` its a built-in package so you don't need to `pip` it. 
I used it to create every **TWO** combinations that can be made up of the input, if you uncomment the `print(x)` you can see how this is stored in `x`. Essentially I've created a list `x` which contains tuples of every two combinations.

Then for each element(tuple) in the list, i pop it and sum them. If they add up to the target then the constieunts are stored in `coffee`.

Its stored as a tuple in list, which is why i use a *hack* to pop the tuple element and make it s list element.

Next I use `list.index("x")` to find out the indices where these numbers belong. This are both stored in `answer`.

Thats it. Now the test cases can be passed. Although it works I admit there is definitely improvements to be made. This wouldn't work for test cases with repeated numbers etc. I also don't have a good route if nothing equals target.

## Method 2
{% highlight python %}
class Solution(object):

    def two_sum(self, nums, target):
        if nums is None or target is None:
            raise TypeError('nums or target cannot be None')
        if not nums:
            raise ValueError('nums cannot be empty')
        cache = {}
        for index, num in enumerate(nums):
            print(cache)
            cache_target = target - num
            if num in cache:
                return [cache[num], index]
            else:
                cache[cache_target] = index
        return None
{% endhighlight %}

The "correct" way to do it as per **donnemartin**. Usual errors are raised for the first two test cases.

I think this is a really clever way to solve this puzzle if you can get your head around it. In summary, a dictionary caled `cache` is initilaised. Then for each index and value of the input, the input and its different to the target is found. 

**What is being stored in cache?**

Good Q! the difference of the current number and its index. why? What we need at the end is two indices. Saving the difference tells us that IF, IF the difference comes up next on the list there was a previous number that can sum to the target. So we only need the stored index, the "future" current index and thats the answer.

I maybe explaining this badly. So, lets try again. We have `x` as the current number, we find the difference between `x` and target. If `x` was 1, and target being 7, the difference would be 7-1=6. This means if 6 was to appear later in the list we know we had a prevoius number that can sum to the target 7. So we only need to store the index of 1, and the difference that can help 1 meet its target.

Let's say 6 does come up in the input. In that case by checking the cache we see 6 with a previous number would meet the target. Since we stored the previous number's index next to 6, we can output that index and the current index(which is 6 in this case) and thats it. VOILA!

## Unit Test

{% highlight python %}
# %load test_two_sum.py
from nose.tools import assert_equal, assert_raises


class TestTwoSum(object):

    def test_two_sum(self):
        solution = Solution()
        assert_raises(TypeError, solution.two_sum, None, None)
        assert_raises(ValueError, solution.two_sum, [], 0)
        target = 7
        nums = [1, 3, 2, -7, 5]
        expected = [2, 4]
        assert_equal(solution.two_sum(nums, target), expected)
        print('Success: test_two_sum')


def main():
    test = TestTwoSum()
    test.test_two_sum()


if __name__ == '__main__':
    main()
{% endhighlight %}














