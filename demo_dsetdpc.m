%This is the code of the DSet-DPC algorithm proposed in
%Jian Hou, Huaqiang Yuan, Marcello Pelillo. Towards Parameter-Free 
%Clustering for Real-World Data. Pattern Recognition, vol. 134, 109062, 2023.

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
        idx_dset=label==num_dsets;
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
