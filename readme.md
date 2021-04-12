# Parallel Programming

## Introduction

### HW1 Even-Odd Sort

In this assignment, we're required to implement the odd-even sort algorithm **using MPI**. Odd-even sort is a comparison sort which consists of two main phases: even-phase and odd-phase. In each phase, processes perform compare-and-swap operations repeatedly as follows until the input array is sorted.

In even-phase, all even/odd indexed pairs of adjacent elements are compared. If the two elements of a pair are not sorted in the correct order, the two elements are swapped. Similarly, the same compare-and-swap operation is repeated for odd/even indexed pairs in odd-phase. The odd-even sort algorithm works by alternating these two phases until the input array is completely sorted.

### HW2 Mandelbrot Set

In this assignment, we're required to implement the mandelbrot set **using Pthread and OpenMP**.

The Mandelbrot Set is a set of complex numbers that are quasi-stable when computed by iterating the function:

![](https://i.imgur.com/rjTQP3t.png)

What exact is the Mandelbrot Set?
* It is fractal: An object that display self-similarity at various scale; magnifying a fractal reveals small-scale details similar to the larger-scale characteristics 
* After plotting the Mandelbrot Set determined by thousands of iterations:
    ![](https://i.imgur.com/U2Luxld.png)

### HW3 All-Pairs Shortest Path (CPU)

In this assignment, we're asked to implement a program that solves the all-pairs shortest path problem.

* We're required to use threading to parallelize the computation in our program.
* We can choose any threading library or framework you like (pthread, std::thread, OpenMP, Intel TBB, etc).
* We can choose any algorithm to solve the problem.
* We must implement the shortest path algorithm ourselves. (Do not use libraries to solve the problem.)

### HW4-1 Blocked All-Pairs Shortest Path (Single-GPU)

In this assignment, we're asked to implement a program that solves the all-pairs shortest path problem **with one GPU card**.

### HW4-2 Blocked All-Pairs Shortest Path (Multi-GPUs)

In this assignment, we're asked to implement a program that solves the all-pairs shortest path problem **with two GPU cards**.