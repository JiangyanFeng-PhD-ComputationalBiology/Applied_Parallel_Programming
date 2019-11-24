# MP7: Sparse Matrix Multiplication 

The purpose of this lab is to implement a SpMV (Sparse Matrix Vector Multiplication) kernel for an input sparse matrix based on the Jagged Diagonal Storage (JDS) transposed format.

## The code performs the following:
Implement sparse matrix-vector multiplication using the JDS format. The kernel shall be launched so that each thread will generate one output Y element. The kernel should have each thread to use the appropriate elements of the JDS data array, the JDS col index array, JDS row index array, and the JDS transposed col ptr array to generate one Y element.

## Authors

Jiangyan Feng - jf8@illinois.edu

Ph.D. Candidate

Chemical and Bimolecular Engineering

University of Illinois at Urbana-Champaign

200 RAL, 600 S. Mathews Ave., Urbana IL 61801
