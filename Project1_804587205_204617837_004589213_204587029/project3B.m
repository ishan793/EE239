clear
clc
A = importdata('E:\UCLA\Quarter2\BigData\Project3\ml-100k\ml-100k\u.data');
k = length(A)/10;
for i=1:10
    test(i,:,:) = get_matrix(A((i-1)*k+1:i*(k),:));
end

