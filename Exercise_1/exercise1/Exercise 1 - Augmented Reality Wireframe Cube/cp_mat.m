function R = cp_mat(k)
    R = [0, -k(3), k(2);
        k(3), 0, -k(1);
        -k(2), k(1), 0];
end