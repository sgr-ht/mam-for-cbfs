We determined to split opcodes that had a space, such as "rep movsd", into two parts, such as "rep" and "movsd", intentionally when we trained fastText. The trained fastText model can generate the vectors of the opcodes using subwords of the opcodes. For example, the vector of "rep movsd" is generated using the subwords of "rep" and "movsd".


This approach appears to be effective because prefixes, such as "rep" and "lock" are combined with multiple opcodes, such as "rep movsb" and "lock add" although this approach changes the vocabulary size of opcodes of fastText.
For example, the number of opcodes in Dataset-1 is 1020, but the vocabulary size of the fastText model is 1018 in this approach.
