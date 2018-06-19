---
layout: post
title: "Compress a String"
date: 2018-03-17 08:05:47
image: '/assets/img/'
description: 'Compressing a string'
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

# String Compression
It is possible to compress a string such that, a string orginally `AAABBCCCDD` becomes `A3B2C3D2`. In thi case we will only compress in order to save space. So for example `AABB` wouldn't be compressed to `A2B2` because it saves no space.


## Tets cases
Let's define some test cases now.

- None -> None
- '' -> ''
- `AABBCC` -> `AABBCC`
- `AAABCCDDDD` -> `A3BC2D4`

## Code
{% highlight python %}
class CompressString(object):

    def compress(self, string):
        if string is None or not string:
            return string
        
        result=''
        prev_char = string[0]
        count = 0 
        
        for char in string:
            if char == prev_char:
                count +=1
            else:
                result+= self.partial_result(prev_char, count)
                prev_char = char
                count = 1
        result += self.partial_result(prev_char, count)
        return result if len(result) < len(string) else string
                
    def partial_result(self, prev_char,count):
        return prev_char + (str(count) if count >1 else '')
{% endhighlight %}

The first thing to check is if the string provided is `None`, if it is, then we can simply return the same thing.

Now the **meat** of the code:
For every character in the given string, if the current character is the same as the last one then add 1 to the count.
If it is not then, save what we have so far(previous letter and its count).
And use the current character as to perform the rest of the loop, if it repeats then increase the count each time.

In the end we check if the compressed string size < string size, if so then we know we are saving space and thus return the compressed string. Or else just return the same string as given.



## Unit Test

{% highlight python %}
from nose.tools import assert_equal


class TestCompress(object):

    def test_compress(self, func):
        assert_equal(func(None), None)
        assert_equal(func(''), '')
        assert_equal(func('AABBCC'), 'AABBCC')
        assert_equal(func('AAABCCDDDDE'), 'A3BC2D4E')
        assert_equal(func('BAAACCDDDD'), 'BA3C2D4')
        assert_equal(func('AAABAACCDDDD'), 'A3BA2C2D4')
        print('Success: test_compress')


def main():
    test = TestCompress()
    compress_string = CompressString()
    test.test_compress(compress_string.compress)


if __name__ == '__main__':
    main()
{% endhighlight %}














