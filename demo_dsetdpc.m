%This is the code of the DSet-DPC algorithm proposed in
%Jian Hou, Huaqiang Yuan, Marcello Pelillo. Towards Parameter-Free Clustering for Real-World Data. Pattern Recognition, vol. 134, 109062, 2023.

function demo_dsetdpc()

    fname='thyroid-2d-tsne.txt';
    descr=dlmread(fname);
    dimen=size(descr,2);
    label_t=descr(:,dimen);
    descr=descr(:,1:dimen-1);
    
    nsample=0.036;
    th_std=0.1;
    
    %build sima
    dima0=pdist(descr,'euclidean');
    dima=squareform(dima0);
    d_mean=mean2(dima);
    sigma=find_sigma(dima,th_std);
    sima=exp(-dima/(d_mean*sigma));
        
    %do clustering
    label_c=dsetpp_extend_dp(sima,dima,nsample);
        
    label_c=reshape(label_c,size(label_t));
    res_nmi=nmi(label_c,label_t)
    res_accuracy=label2accuracy(label_c,label_t)

end

function label=dsetpp_extend_dp(sima,dima,nsample)

    toll=1e-4;
    ndata=size(sima,1);
    label=zeros(1,ndata);
    nsample=max(round(ndata*nsample),5);

    %dp data
    rho=zeros(1,ndata);
    for i=1:ndata
        vec=sima(i,:);
        vec1=sort(vec,'descend');
        vec_descend=vec1(2:nsample+1);
        rho(i)=mean(vec_descend);
    end

    [~,~,nneigh]=find_delta(rho,dima);
    
    min_size=3;
    th_size=min_size+1;             %the minimum size of a cluster
            
    %dset initialization
    for i=1:ndata
        sima(i,i)=0;
    end
    x=ones(ndata,1)/ndata;
    
    %start clustering
    num_dsets=0;
    while 1>0
        if sum(label==0)<5
            break;
        end

        %dset extraction
        x=indyn(sima,x,toll);
        idx_dset=find(x>0);

        if length(idx_dset)<th_size
            break;
        end

        num_dsets=num_dsets+1;
        label(idx_dset)=num_dsets;
        
        %expansion by dp
        label=cluster_extend_dp(nneigh,dima,label,num_dsets);
        
        %post-processing
        idx=label>0;
        sima(idx,:)=0;
        sima(:,idx)=0;

        idx_ndset=find(label==0);
        num_ndset=length(idx_ndset);
        x=zeros(ndata,1);
        x(idx_ndset)=1/num_ndset;
    end
    
end

function label=cluster_extend_dp(nneigh,dima,label,num_dsets)

    %start extension
    while 1>0
        idx_ndset=find(label==0);
        idx_dset=find(label==num_dsets);
        sub_dima=dima(idx_ndset,idx_dset);
        [~,idx_min]=min(sub_dima,[],1);
        idx_out=idx_ndset(idx_min);
        idx_out=unique(idx_out);

        flag=0;
        for i=idx_out
            idx_neigh=nneigh(i);
                        
            if idx_neigh==0
                continue;
            end
            
            if label(idx_neigh)==num_dsets
                label(i)=num_dsets;
                flag=1;
            end
        end
        
        if flag==0
            break;
        end
    end
    
end

function sigma=find_sigma(dima,th_std)

    dmean=mean(dima(:));

    sigma=1;
    for sigma0=1:10
        sima=exp(-dima/(dmean*sigma0));
        
        tri=triu(sima,1);
        v_tri=tri(:);
        v_sima=v_tri(v_tri>0);
        st=std(v_sima);
        
        if st<th_std
            sigma=sigma0;
            break;
        end
    end

end

function [delta,ordrho,nneigh]=find_delta(rho,dist)

    ND=length(rho);
    delta=zeros(1,ND);
    nneigh=zeros(1,ND);
    
    maxd=max(max(dist));
    
    [~,ordrho]=sort(rho,'descend');
    delta(ordrho(1))=-1;
    nneigh(ordrho(1))=0;
    
    for i=2:ND
        delta(ordrho(i))=maxd+1;
        for j=1:i-1
            if(dist(ordrho(i),ordrho(j))<delta(ordrho(i)))
                delta(ordrho(i))=dist(ordrho(i),ordrho(j));
                nneigh(ordrho(i))=ordrho(j);
            end
        end
    end
    delta(ordrho(1))=max(delta(:));
    
end

function [descr, label, fname, name_data]=clusterdata_load(idx)

    name_data=cell(1,120);
    name_data(1)={'d31.txt'};            %3100 * 2 * 31
    name_data(2)={'r15.txt'};            %600 * 2 * 15
    name_data(3)={'unbalance.txt'};     %6500 * 2 * 8
    name_data(4)={'varydensity.txt'};   %150 * 2 * 3
    name_data(5)={'s1.txt'};            %5000 * 2 * 15
    name_data(6)={'s2.txt'};            %5000 * 2 * 15
    name_data(7)={'a1.txt'};            %3000 * 2 * 20
    name_data(8)={'a2.txt'};            %5250 * 2 * 35
    name_data(9)={'a3.txt'};            %7500 * 2 * 50
    name_data(10)={'dim032.txt'};        %1024 * 32 * 16
    name_data(11)={'dim064.txt'};        %1024 * 64 * 16
    name_data(12)={'dim128.txt'};        %1024 * 128 * 16
    name_data(13)={'dim256.txt'};        %1024 * 256 * 16
    name_data(14)={'dim512.txt'};        %1024 * 512 * 16
    name_data(15)={'dim1024.txt'};       %1024 * 1024 * 16
    name_data(16)={'spread-2-10.txt'};   %1000 * 2 * 10
    name_data(17)={'spread-10-20.txt'};  %2000 * 10 * 20
    name_data(18)={'spread-20-35.txt'};  %3500 * 20 * 35
    name_data(19)={'spread-35-2.txt'};   %200 * 35 * 2
    name_data(20)={'spread-50-50.txt'};  %5000 * 50 * 50
    
    name_data(21)={'thyroid.txt'};        %215 * 5 * 3
    name_data(22)={'wine.txt'};           %178 * 13 * 3
    name_data(23)={'iris.txt'};           %150 * 4 * 3
    name_data(24)={'glass.txt'};          %214 * 9 * 6
    name_data(25)={'wdbc.txt'};           %569 * 30 * 2
    name_data(26)={'breast.txt'};         %699 * 9 * 2
    name_data(27)={'leaves.txt'};         %1600 * 64 * 100
    name_data(28)={'segment.txt'};        %2310 * 19 * 7
    name_data(29)={'libras.txt'};         %360 * 90 * 15
    name_data(30)={'ionosphere.txt'};     %351 * 34 * 2
    name_data(31)={'waveform.txt'};       %5000 * 21 * 3
    name_data(32)={'waveform_noise.txt'}; %5000 * 40 * 3
    name_data(33)={'ecoli.txt'};          %336 * 7 * 8
    name_data(34)={'cnae9.txt'};          %1080 * 856 * 9
    name_data(35)={'Olivertti.txt'};      %400 * 28 * 40
    name_data(36)={'dermatology.txt'};    %366 * 33 * 6
    name_data(37)={'balance-scale.txt'};  %625 * 4 * 3
    name_data(38)={'robotnavi.txt'};      %5456 * 24 * 4
    name_data(39)={'scc.txt'};            %600 * 60 * 6
    name_data(40)={'usps.txt'};           %11000 * 256 * 10

    direc='2d\';
%     if idx<10 || idx==16
%         fname=[direc,name_data{idx}];
%     else
        sname=name_data{idx};
        sname=sname(1:length(sname)-4);
        sname=[sname,'-2d-tsne.txt'];
        fname=[direc,sname];
%     end
    descr=dlmread(fname);
    dimen=size(descr,2);
    label=descr(:,dimen);
    descr=descr(:,1:dimen-1);

end

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

function z = nmi(x, y)
% Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
% Input:
%   x, y: two integer vector of the same length 
% Ouput:
%   z: normalized mutual information z=I(x,y)/sqrt(H(x)*H(y))
% Written by Mo Chen (sth4nth@gmail.com).
assert(numel(x) == numel(y));
n = numel(x);
x = reshape(x,1,n);
y = reshape(y,1,n);

l = min(min(x),min(y));
x = x-l+1;
y = y-l+1;
k = max(max(x),max(y));

idx = 1:n;
Mx = sparse(idx,x,1,n,k,n);
My = sparse(idx,y,1,n,k,n);
Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
Hxy = -dot(Pxy,log2(Pxy));

% hacking, to elimative the 0log0 issue
Px = nonzeros(mean(Mx,1));
Py = nonzeros(mean(My,1));

% entropy of Py and Px
Hx = -dot(Px,log2(Px));
Hy = -dot(Py,log2(Py));

% mutual information
MI = Hx + Hy - Hxy;

% normalized mutual information
z = sqrt((MI/Hx)*(MI/Hy));
z = max(0,z);

end

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
