function [gbestX,gbestfitness,gbesthistory] = OSCOA(H,popsize,dim,xmax,xmin,vmax,vmin ,MaxFEs,func,fid,~)

IterFEs=popsize+ceil(popsize/20);  
maxiter=ceil(MaxFEs/IterFEs); 
overFEs = maxiter*IterFEs;
gbesthistory=rand(1,overFEs);  
FEsfitness = inf;
FEs = 0;


costs=rand(1,popsize);
pop = repmat(xmin,popsize,1)+(xmax-xmin).*rand(popsize,dim);
for z=1:popsize
    costs(z) = func(pop(z,:)',fid);  
    FEs=FEs+1;
    FEsfitness=min([FEsfitness,costs(z)]);
    gbesthistory(FEs)=FEsfitness;
end

ages = ones(1,popsize);

year =0;
n_packs = 10;
Ps  = 1/dim;
iwt = 0.9-(0.6.*(1:MaxFEs)/MaxFEs);

[GlobalMin,bestX_pos]=min(costs);
while FEs<=MaxFEs
    
    rand_coy = randperm(popsize);
    %%  FDC
   
    distance =pdist2(pop(bestX_pos,:),pop);  
    %% Fitness-distance correlation(FDC)
    det_costs = costs-GlobalMin;      
    mean_dis=mean(distance);          
    mean_costs = mean(det_costs);      
    std_dis = std(distance);           
    std_costs = std(det_costs);
    C_u = sum((distance-mean_dis).*(det_costs-mean_costs))/popsize;
    if std_dis==0 || std_costs==0
        FDC = 0;
    else
        FDC = C_u/(std_dis*std_costs);
    end
    %% Subpopulation size
    if rand>FDC
        if n_packs<15  
            n_packs=n_packs+1;
        end
    else                
        if n_packs>10
            n_packs=n_packs-1;
        end
    end
    year = year+1;
    
    n_coy = floor(popsize/n_packs); 
    rem_coys = rem(popsize,n_packs);  
    %Allocate remaining coyotes to n_packs of subpopulations
    pack_end = 0;
    for j = 1:n_packs
        pack_head = pack_end+1;
        pack_end = pack_head+n_coy-1;
        if j<=rem_coys
            pack_end = pack_end+1;
        end
        pack = rand_coy(pack_head:pack_end); %Subpopulation Index
        
        coyotes_aux = pop(pack,:);
        costs_aux = costs(pack);
        ages_aux = ages(pack);
        n_coy_aux =length(pack);
        
        [costs_aux,inds]=sort(costs_aux,'ascend');
        coyotes_aux      = coyotes_aux(inds,:);
        ages_aux         = ages_aux(inds);
        c_alpha          = coyotes_aux(1,:);  
        tendency         = median(coyotes_aux,1); 
        new_coyotes      = zeros(n_coy_aux,dim);
        for c=1:n_coy_aux
            rc1 = c;
            while rc1==c
                rc1 = randi(n_coy_aux);  
            end
            rc2 = c;
            while rc2==c || rc2 == rc1
                rc2 = randi(n_coy_aux);  
            end
            %% Parameter selection
            if rand>FDC   
                ww = rand(1,4); 
                ww(1:2) =-2+rand(1,2);  %[-2:-1]
                ww(3:4) = 1+rand(1,2); %[1:-2]
                w= ww(randperm(4,2));
                new_c = coyotes_aux(c,:) + w(1)*rand*(c_alpha - coyotes_aux(rc1,:))+ ...
                    w(2)*rand*(tendency  - coyotes_aux(rc2,:));
            else 
                PP2 = PP2+1;
                if FEs<=MaxFEs
                    w = [iwt(FEs)];
                else
                    w=[iwt(MaxFEs)];
                end
                new_c = coyotes_aux(c,:) + w(1)*rand*(c_alpha - coyotes_aux(rc1,:));
            end
            
            
            new_c = min(max(new_c,xmin),xmax); %Cross-border detection
            
            new_cost = func(new_c',fid);
            FEs=FEs+1;
            
            FEsfitness=min([FEsfitness,new_cost]);
            gbesthistory(FEs)=FEsfitness;
            
            new_coyotes(c,:)=new_c;
          
            if new_cost<costs_aux(c)
                costs_aux(c)      = new_cost;
                coyotes_aux(c,:)    = new_coyotes(c,:);
            end
        end
        
        %% Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
        if rand>FDC  
            parents = randperm(n_coy_aux,2);      
        else 
            parents = [1,randperm(n_coy_aux,1)];
        end
        
        prob1           = (1-Ps)/2;
        prob2           = prob1;
        pdr             = randperm(dim);
        p1              = zeros(1,dim);
        p2              = zeros(1,dim);
        p1(pdr(1))      = 1; % Guarantee 1 charac. per individual
        p2(pdr(2))      = 1; % Guarantee 1 charac. per individual
        r               = rand(1,dim-2);
        p1(pdr(3:end))  = r < prob1;
        p2(pdr(3:end))  = r > 1-prob2;

        n  = ~(p1|p2);

        % Generate the pup considering intrinsic and extrinsic influence
        pup =   p1.*coyotes_aux(parents(1),:) + ...
            p2.*coyotes_aux(parents(2),:) + ...
            n.*(xmin + rand(1,dim).*(xmax-xmin));
        
        % Verify if the pup will survive
        pup_cost    = func(pup',fid);
        FEs=FEs+1;
        FEsfitness=min([FEsfitness,pup_cost]);
        gbesthistory(FEs)=FEsfitness;
        
        
        worst       = find(pup_cost<costs_aux==1);
        if ~isempty(worst)
            [~,older]               = sort(ages_aux(worst),'descend');
            which                   = worst(older);
            coyotes_aux(which(1),:) = pup;
            costs_aux(which(1))   = pup_cost;
            ages_aux(which(1))    = 0;
        end
        
        %% Update the pack information
        pop(pack,:) = coyotes_aux;
        costs(pack)   = costs_aux;
        ages(pack)    = ages_aux;
    end  
    
    %% Update coyotes ages
    ages = ages + 1;
    
    [GlobalMin,bestX_pos]=min(costs);
    gbestX = pop(bestX_pos,:);
 
    
end
gbesthistory=gbesthistory(1:MaxFEs);
gbestfitness = gbesthistory(MaxFEs);
end