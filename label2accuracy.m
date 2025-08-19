function rate=label2accuracy(label_c,label_t)

    ndata=length(label_c);
    matr_c=zeros(ndata);
    matr_t=zeros(ndata);
    
    nsame=0;
    for i=1:ndata
        for j=1:ndata
            if label_c(i)==label_c(j)
                matr_c(i,j)=1;
            end
            if label_t(i)==label_t(j)
                matr_t(i,j)=1;
            end
            
            if matr_c(i,j)==matr_t(i,j)
                nsame=nsame+1;
            end
        end
    end
    
    rate=nsame/(ndata*ndata);

end