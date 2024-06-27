% Compute "representativeness" predictions for the preference learning model
%
% Steps
% 1. Sample N sets of utility from prior
% 2. Condition on u_x being highest by filtering out all other prior samples
% 3. For each choice, compute p(choose option w/ x | u) for all u's
% 4. Compute the mean of this distribution to approximate
% 5. Then ranking \propto E( p(choose option w/ x | u) )
% 6. Convert to fractional rankings
%

close all;
clear all;

pp = parpool;


%
% We assume a gaussian prior on utilities such that ~98% of the mass lies above
% 0, i.e. mean = 2 * std
s_u = 2; 
u_sign = -1; % Whether utilities are positive (1) or negative (-1)

% And we assume the selection is made using the following probabilistic rule
%   P(i) = exp(u_i)^(1/s) / (\sum_j{exp(u_j)^(1/s)})
% When s -> 0, choice rule becomes maximum utility
% When s -> 1, choice rule becomes utility matching (probabilistic selection)
% When s -> inf, choice rule becomes random
s = 1;

DIFFERENCE_THRESHOLD = 5e-4;

nsamples = 20000000;

% X{i} = a [option-by-effect] matrix where each row represents x_i
% Effects: [d c b a x]

X{1} =  [0 0 0 1 0;
         0 0 0 0 1];
labels{1} = 'a/x';
X{2} =  [1 1 1 0 0;
         0 0 0 1 1];
labels{2} = 'dcb/ax';
X{3}  = [1 0 0 0 1;
         0 1 0 0 1;
         0 0 1 0 1;
         0 0 0 1 1];
labels{3} = 'dx/cx/bx/ax';
X{4} =  [0 0 1 1 0;
         0 0 0 1 1];
labels{4} = 'ab/ax';
X{5}  = [1 1 1 1 0;
         0 0 0 0 1];
labels{5} = 'dcba/x';
X{6} =  [1 1 1 0 0;
         0 0 0 1 0;
         0 0 0 0 1];
labels{6} = 'dcb/a/x';
X{7} =  [1 1 0 0 1;
         0 0 1 1 1];
labels{7} = 'dcx/bax';
X{8}  = [1 0 1 0 1;
         0 1 0 1 1;
         0 0 1 1 1];
labels{8} = 'bdx/cax/bax';
X{9}  = [1 0 1 1 0;
         0 1 1 0 1;
         0 0 1 1 1];
labels{9} = 'bad/bcx/bax';
X{10} = [1 1 0 0 0;
         0 0 1 1 0;
         0 0 0 0 1];  
labels{10} = 'dc/ba/x';                
X{11} = [1 0 0 0 0;
         0 1 0 0 0;
         0 0 1 0 0;
         0 0 0 1 0;
         0 0 0 0 1];   
labels{11} = 'd/c/b/a/x';    
X{12} = [1 1 1 0 0;
         0 0 1 1 1];     
labels{12} = 'bdc/bax';
X{13} = [0 1 1 0 0;
         0 0 0 1 0;
         0 0 0 0 1];
labels{13} = 'cb/a/x';
X{14} = [0 0 1 1 1];
labels{14} = 'bax';
X{15} = [1 0 1 0 1;
         0 1 1 0 1;
         0 0 1 1 1];
labels{15} = 'bdx/bcx/bax';
X{16} = [0 1 1 0 0;
         0 0 0 1 1];
labels{16} = 'cb/ax';
X{17} = [1 0 0 1 0;
         0 1 0 1 0;
         0 0 1 0 1;
         0 0 0 1 1];
labels{17} = 'ad/ac/bx/ax';
X{18} = [1 0 0 0 0;
         0 1 0 0 0;
         0 0 1 1 1];
labels{18} = 'd/c/bax';
X{19} = [0 0 0 0 1];   
labels{19} = 'x';                            
X{20} = [1 1 0 0 0;
         0 0 1 1 1];
labels{20} = 'dc/bax';
X{21} = [0 0 1 0 0;
         0 0 0 1 0;
         0 0 0 0 1];    
labels{21} = 'b/a/x';                       
X{22} = [1 1 0 0 0;
         0 0 1 0 0;
         0 0 0 1 0;
         0 0 0 0 1];  
labels{22} = 'dc/b/a/x';    
X{23} = [1 0 0 1 0;
         0 1 0 0 1;
         0 0 1 0 1;
         0 0 0 1 1];
labels{23} = 'ad/cx/bx/ax';
X{24} = [0 1 0 1 0;
         0 0 1 0 1;
         0 0 0 1 1];       
labels{24} = 'ac/bx/ax';
X{25} = [1 1 0 0 0;
         0 0 1 0 0;
         0 0 0 1 1];    
labels{25} = 'dc/b/ax';            
X{26} = [0 0 0 1 1];
labels{26} = 'ax';
X{27} = [0 1 0 0 0;
         0 0 1 0 0;
         0 0 0 1 1];
labels{27} = 'c/b/ax';     
X{28} = [1 1 0 1 1;
         0 1 1 1 1];
labels{28} = 'dcax/bcax';
X{29} = [0 1 0 1 0;
         0 0 1 1 0;
         0 0 0 1 1];     
labels{29} = 'ac/ab/ax';            
X{30} = [1 1 0 0 0;
         0 0 1 0 1;
         0 0 0 1 1];
labels{30} = 'dc/bx/ax';
X{31} = [0 0 1 1 0;
         0 0 0 0 1];    
labels{31} = 'ba/x';    
X{32} = [0 0 1 0 1;
         0 0 0 1 1];
labels{32} = 'bx/ax';
X{33} = [0 1 0 0 1;
         0 0 1 0 1;
         0 0 0 1 1]; 
labels{33} = 'cx/bx/ax';                
X{34} = [1 0 0 0 0;
         0 1 0 0 0;
         0 0 1 0 0;
         0 0 0 1 1];    
labels{34} = 'd/c/b/ax';            
X{35} = [0 0 1 0 0;
         0 0 0 1 1];
labels{35} = 'b/ax';
X{36} = [0 1 0 0 0;
         0 0 1 1 1];
labels{36} = 'c/bax';       
X{37} = [0 1 1 1 0;
         0 0 0 0 1];
labels{37} = 'cba/x';
X{38} = [0 1 0 1 1;
         0 0 1 1 1]; 
labels{38} = 'cax/bax';
X{39} = [0 1 1 1 1];
labels{39} = 'cbax';
X{40} = [1 1 1 1 1];
labels{40} = 'dcbax';
X{41} = [1 1 1 1 0;
         0 1 1 1 1];
labels{41} = 'cbad/cbax';
X{42} = [1 0 1 1 0;
         0 0 1 1 1];
labels{42} = 'bad/bax';
X{43} = [1 0 0 1 0;
         0 1 0 1 0;
         0 0 1 1 0;
         0 0 0 1 1];
labels{43} = 'ad/ac/ab/ax';
X{44} = [1 1 0 0 0;
         0 0 1 1 0;
         0 0 0 1 1];
labels{44} = 'dc/ab/ax';
X{45} = [0 1 0 0 0;
         0 0 1 0 0;
         0 0 0 1 0;
         0 0 0 0 1];
labels{45} = 'c/b/a/x';
X{46} = [1 0 0 0 0;
         0 1 1 1 1];   
labels{46} = 'd/cbax';               
X{47} = [1 0 1 1 0;
         0 1 1 1 0;
         0 0 1 1 1];
labels{47} = 'bad/bac/bax';

nproblems = 47;
n = 5; % Num of effects 5
run_sampler = 1;

    if (run_sampler == 1)
        
        % Draw samples from the prior
        if (u_sign > 0)
            priorsamples = mvnrnd(zeros(1,n)+2*s_u, eye(n,n)*s_u, nsamples);
        else
            priorsamples = mvnrnd(zeros(1,n)-2*s_u, eye(n,n)*s_u, nsamples);
        end
        
        % Filter out samples where x isn't highest
        nFilteredSamples = 0;
        for i=1:nsamples
            if (mod(i,1000000) == 0)
                fprintf('Pre-scan sample %d\n',i);
            end
            if (max(priorsamples(i,:)) == priorsamples(i,5))
                nFilteredSamples = nFilteredSamples+1;
            end
        end
        filteredsamples = zeros(nFilteredSamples,5);
		sampleIndex = 1;
        for i=1:nsamples
            if (mod(i,1000000) == 0)
                fprintf('Filter sample %d\n',i);
            end
            if (max(priorsamples(i,:)) == priorsamples(i,5))
                filteredsamples(sampleIndex,:) = priorsamples(i,:);
				sampleIndex = sampleIndex+1;
            end
        end
    
        parfor problem=1:nproblems
            
            fprintf('=== Problem %d ===\n', problem);
            
            likelihoods{problem} = zeros(1,nFilteredSamples);
            
            for i=1:nFilteredSamples
            
                if (mod(i,1000000) == 0)
                    fprintf('Iteration %d\n',i);
                end
                
                % Get current sample
                y = filteredsamples(i,1:n);
                
                % Compute likelihood
                likelihoods{problem}(i) = exp(y * X{problem}(end,:)').^(1/s) / sum(exp(y * X{problem}').^(1/s));
                                                     
            end
        end
        
        % Compute expected values
        for problem = 1:nproblems
            % E(choose X)
            pChoose(problem) = mean(likelihoods{problem});
        end
        
    
    end
    
    % Compute the rankings
    
    % 4) P(choose option w/ X | X is highest)
    sortedRankingRep = zeros(1,nproblems);
    [sortedRep, sortingIndex] = sort(pChoose);
    
    % Loop through and collect a group of items that are not significantly different
    % from one another
    currRank = 1;
    eqClass = [1];
    for p=2:nproblems
        if (abs(sortedRep(p) - sortedRep(p-1)) > DIFFERENCE_THRESHOLD)
            % Record the fractional rank of all items in the equivalence class
            fractionalRank = sum(currRank:(currRank+length(eqClass)-1)) / length(eqClass);
            sortedRankingRep(eqClass) = fractionalRank;
            % Increment the current rank
            currRank = p;
            % Reset the eqivalence class
            eqClass = [p];
        else
            eqClass = [eqClass p];
        end
    end
    % Dump the remaining equivalence class
    fractionalRank = sum(currRank:(currRank+length(eqClass)-1)) / length(eqClass);
    sortedRankingRep(eqClass) = fractionalRank;
    % Now "invert" the ranking vector so they are ordered by problem number
    rankingRep(sortingIndex) = sortedRankingRep;
    
    
    % Save the results
    %save('representativenessmodelpredictions','pChoose','labels','rankingRep');

delete(pp);


