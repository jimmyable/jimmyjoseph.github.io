---
layout: post
title: "AI TicTacToe"
date: 2017-10-20 03:32:44
image: '/assets/img/'
description: 'A simple game modelled in Python with a computer "AI"'
tags:
- python
- AI
- game
categories:
- ATBSWP
twitter_text:
---

## Basic Setup

1. [Install Python 3](https://www.python.org/)
2. Clone the [repo](https://github.com/jimmyjoseph1295/AITicTacToe)
3. Run the `game.py` file

## Further Support

If you don't know how Python works, this code was inspired `heavily` from [Al Sweigart's](https://inventwithpython.com/chapter10.html) resources. He provides very detailed and beginner friendly python books and they're all free.

## Are you still here?

Have a code snippet then...

{% highlight python %}
def playAgain():
# This function returns True if the player wants to play again, otherwise it returns False.
	print('Do you want to play again? (yes or no)')
	return input().lower().startswith('y')

{% endhighlight %}






