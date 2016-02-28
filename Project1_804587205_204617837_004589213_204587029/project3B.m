clear
clc
A = importdata('E:\UCLA\Quarter2\BigData\Project3\ml-100k\ml-100k\u.data');

%code for part A
for i=1:100000
    M(A(i,1),A(i,2))= A(i,3);
end

[A,Y,numIter,tElapsed,finalResidual] = wnmfrule(M,10);

% code for part B
k = length(A)/10;
for i=1:10
    test(i,:,:) = get_matrix(A((i-1)*k+1:i*(k),:));
end

