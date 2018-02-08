---
layout: post
title: "Basic Java"
date: 2018-02-08 21:34:47
image: '/assets/img/'
description: 'Exploring Java language and its basics'
tags:
- java
- loops
- stacks
- queues
- compile
categories:
- BasicJava
twitter_text: 
---

## Why Java?


Java is old, older than me atleast. But its not seen as old, its seen as a standard programming language. If you're learning how to program you will end up with java at some point. And for good reason, its an efficient program that has standed the test of time. 


In my opinion it is more "messy" for a beginner, you need to write more code. Mainly because everything is explicit and the language is designed for high end efficient softwares. Unlike python which uses much simpler scripting language but is more suited for quick "prototyping". 

It is recommended to use an IDE like [Eclipse](https://www.eclipse.org/) or [IntelliJ IDEA](https://www.jetbrains.com/idea/) if you plan to actually write java code. Missing paranthesis or brackets or simple mistakes can mean wasting a lot of time looking for what is going wrong after compiling. 

## Installing Java
There are many ways to install and versions change all the time, for ubuntu try:

{% highlight bash %}
username@hostname:~$ sudo apt-get install openjdk-8-jdk
{% endhighlight %}

## Running Java code
You have to [compile](https://www.youtube.com/watch?v=G1ubVOl9IBw) a java file into a class file before you can run the code. In bash you can compile a File.java like this:

{% highlight bash %}
username@hostname:~$ javac filepath/File.java
{% endhighlight %}

Then you can run the compiled `.class` file as follows. Compiled means its code that your computer can understand and execute easily.

{% highlight bash %}
username@hostname:~$ java filepath/File
{% endhighlight %}

## How to write "Hello World"
Yeah its like five lines. This looks daunting to a beginner but you're just going to have to not understand some terms like `public class` or `main`. The quick overview is that every java file has classes, public and private, each class can have methods and every java code needs atleast one `main` method that runs the rest of the code.

The "print function" is the `System.out.println`, for now imagine the rest as scaffolding.

{% highlight java %}
public class HelloWorld {
  public static void main(String[] args){
    System.out.println("Hello World!");
  }
}
{% endhighlight %}

## Taking Inputs from command line

When you run a java code you cna send in intial input that the code can work with these are called `arguments`. We will also import a Scanner module to "scan" through the inputs. Why am i writing scanner like three times? Because you have to. I'm saying 'scanner' is a "Scanner" object, then im creating a new Scanner object with the input fed. You can ue scanner to take in more arguments and even make simple question answering games.


{% highlight java %}
import java.util.Scanner;

public class Input {
  public static void main(String[] args){
  print_input();
  }

  private static void print_input() {
    System.out.println("I'm a parrot, tell me something");
    Scanner scanner =  new Scanner(System.in);
    String usaid = scanner.next();
    System.out.println("PARROT NOISE "+ usaid);
  }
}
{% endhighlight %}

## Performing Operations
We can take in two numbers from the commandline and perform some operations. Here is the code for it below:

{% highlight java %}
public class ValueOfDemo{
  public static void main(String[] args){
    //check if there are atleast 2 arguments
    if (args.length == 2) {
      //convert strings to numbers
      float a = (Float.valueOf(args[0])).floatValue();
      float b = (Float.valueOf(args[1])).floatValue();

      //do some arithemetic
      System.out.println("a +b = " + (a + b));
      System.out.println("a - b = "+ (a-b));
      System.out.println("a * b = "+ (a*b));
      System.out.println("a / b = "+ (a/b));
    } else {
      System.out.println("This program needs two or more command line args");
    }
  }
}
{% endhighlight %}
This file would be saved as "ValueOfDemo.java" and you would run it as follows:
{% highlight bash %}
username@hostname:~$ javac filepath/ValueOfDemo.java
username@hostname:~$ java filepath/ValueOfDemo 5 10
{% endhighlight %}

## Loops
The different simplest loops you can do in Java are as follows:
- For loops
- While loops
- Do while loops
- String loops


{% highlight java %}
public class Loops {
  public static void main(String[] args) {
    System.out.println("Working with looooops");
    //for_loop();
    //enhanced_for_loop();
    //string_loop();
    //while_loop();
    do_while_loop();
  }
  public static void for_loop() {
    System.out.println("Here be the for loop");
    for(int i = 0; i < 10; i++) {
      System.out.println(i);
    }
  }

  private static void enhanced_for_loop() {
    System.out.println("This is the enhanced for loop");
    int[] numbers = {0,1,2,3,4,5,6,7,8,9};
    for(int n : numbers) {
        ///extend by agggregating a list and print.out every loop
      System.out.println(n);
    }
  }

  private static void string_loop() {
    System.out.println("This is a loop of strings");
    String s = new String("this be my string!");
    for(int i = 0; i < 10; i++) {
      System.out.println(s+" iteration = "+i);
    }
  }
  private static void while_loop() {
    System.out.println("This is a while loop");
    int n = 0;
    while(n < 10) {
      System.out.println(n);
      n++;
    }
  }

  private static void do_while_loop() {
    System.out.println("This is a do while loop");
    int n = 0;
    do {
      System.out.println(n);
      n++;
    } while(n < 20);
  }
}
{% endhighlight %}


## Stacks
Imagine a [stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type)) of paper. You add another sheet. If you were asked to take a sheet off, you'd pick the last sheet you placed. This is how a stack works, Last In First Out(`LIFO`), its a type of queue. It can be implemented as follows:

{% highlight java %}
public class Loops {
  public static void main(String[] args) {
    System.out.println("Working with looooops");
    //for_loop();
    //enhanced_for_loop();
    //string_loop();
    //while_loop();
    do_while_loop();
  }
  public static void for_loop() {
    System.out.println("Here be the for loop");
    for(int i = 0; i < 10; i++) {
      System.out.println(i);
    }
  }

  private static void enhanced_for_loop() {
    System.out.println("This is the enhanced for loop");
    int[] numbers = {0,1,2,3,4,5,6,7,8,9};
    for(int n : numbers) {
        ///extend by agggregating a list and print.out every loop
      System.out.println(n);
    }
  }

  private static void string_loop() {
    System.out.println("This is a loop of strings");
    String s = new String("this be my string!");
    for(int i = 0; i < 10; i++) {
      System.out.println(s+" iteration = "+i);
    }
  }
  private static void while_loop() {
    System.out.println("This is a while loop");
    int n = 0;
    while(n < 10) {
      System.out.println(n);
      n++;
    }
  }

  private static void do_while_loop() {
    System.out.println("This is a do while loop");
    int n = 0;
    do {
      System.out.println(n);
      n++;
    } while(n < 20);
  }
}
{% endhighlight %}

## Queue
And heres how to do a normal queue, you know first come first served in a resturant? or First In First Out(`FIFO`):
{% highlight java %}
import java.util.Vector;

public class Queue {
  private static Vector<Object> queue =  new Vector<Object>();
  public static void main(String[] args) {
    enqueue("dequeue should return the first element pushed in");
    enqueue("now the second");
    enqueue("third");
    enqueue("you get the idea");
    System.out.println(dequeue());
    System.out.println(dequeue());
    System.out.println(dequeue());
    System.out.println(dequeue());

  }

  public static boolean empty() {
    return queue.isEmpty();
  }

  public static Object dequeue() {
    return queue.remove(0);
  }

  public static void enqueue(Object obj) {
    queue.add(obj);
  }
}
{% endhighlight %}

## Extras
If you want to do more all this code and more is on my [BasicJava](https://github.com/jimmyjoseph1295/BasicJava) repo. Theres also an exercise that created a Mortage calculator. 






