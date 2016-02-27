function res = get_matrix(ip_data)
    res = zeros(943,1682);
    for i = 1:length(ip_data)
        res(ip_data(i,1),ip_data(i,2))=ip_data(i,3);
    end
end