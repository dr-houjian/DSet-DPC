%This function is used to extract a dominant set from a similarity matrix
%A, with x as the initial state, and toll as the error limit
%from S.R. Bulo, M. Pelillo, I.M. Bomze, Graph-based quadratic optimization: a
%fast evolutionary approach, Comput. Vis. Image Understand. 115 (7) (2011)
%984¨C995 .
%written by S. R. Bulo
function x=indyn(sima,x,toll)

    dsima=size(sima,1);
    if (~exist('x','var'))
        x=zeros(dsima,1);
        maxv=max(sima);
        for i=1:dsima
            if maxv(i)>0
                x(i)=1;
                break;
            end
        end
    end
    
    if (~exist('toll','var'))
        toll=0.005;
    end
    
    for i=1:dsima
        sima(i,i)=0;
    end
    
    x=reshape(x,dsima,1);

    %start operation
    g = sima*x;
    AT = sima;
    h = AT*x;
    niter=0;
    while 1
        r = g - (x'*g);
        
        if norm(min(x,-r))<toll
            break;
        end
        i = selectPureStrategy(x,r);
        den = sima(i,i) - h(i) - r(i); %In case of asymmetric affinities
        do_remove=0;
        if r(i)>=0
            mu = 1;
            if den<0
                mu = min(mu, -r(i)/den);
                if mu<0 
                    mu=0; 
                end
            end
        else
            do_remove=1;
            mu = x(i)/(x(i)-1);
            if den<0
                [mu,do_remove] = max([mu -r(i)/den]);
                do_remove=do_remove==1;
            end
        end
        tmp = -x;
        tmp(i) = tmp(i)+1;
        x = mu*tmp + x;
        if(do_remove) 
           x(i)=0; 
        end
        x=abs(x)/sum(abs(x));
        
        g = mu*(sima(:,i)-g) + g;
        h = mu*(AT(:,i)-h) + h; %In case of asymmetric affinities
        niter=niter+1;
    end
    
    x=x';
end

function [i] = selectPureStrategy(x,r)
    index=1:length(x);
    mask = x>0;
    masked_index = index(mask);
    [~, i] = max(r);
    [~, j] = min(r(x>0));
    j = masked_index(j);
    if r(i)<-r(j)
        i = j;
    end
    return;
end