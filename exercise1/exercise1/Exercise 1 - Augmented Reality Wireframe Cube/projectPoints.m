function Pp=projectPoints(T,K,pt_pos_w)
    Pp = K*T*[pt_pos_w,1]';
    Pp = Pp/Pp(3);
end