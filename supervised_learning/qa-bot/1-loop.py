#!/usr/bin/env python3

terminators = ['exit', 'quit', 'goodbye', 'bye']
print("Q: ", end='')
q = input()

while q.lower() not in terminators:
    print('A:')
    print("Q: ", end='')
    q = input()

print("A: Goodbye")