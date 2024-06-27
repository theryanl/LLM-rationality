% We assume that there is a vector of decisions d = [d_1 d_2 ... d_m]
% d_i = {1,2,3,...,c} where c is the number of choices.
% There are n possible effects for each decision.
%
% x_i is a binary vector indicating which of the m effects follow from
% decision d_i.
%
% beta is a vector of utilities assigned to each of the effects, where
% we assume that the utility for a decision d_i is the sum of the
% utilities of its effects.
%
% Therefore, P(beta | d,X) \propto P(d | beta,X) * P(beta | X)
%                          \propto P(d | beta,X) * P(beta)
%
%

close all;
clear all;

pp = parpool;


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

% How many importance samples to draw
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
    
        parfor problem=1:nproblems
            
            fprintf('=== Problem %d ===\n', problem);
            
            w{problem} = zeros(1,nsamples);
            w_max{problem} = zeros(1,nsamples);
            
            for i=1:nsamples
            
                if (mod(i,1000000) == 0)
                    fprintf('Iteration %d\n',i);
                end
                
                % Get current sample
                y = priorsamples(i,1:n);
                
                % Compute w term for full model
                w{problem}(i) = exp(y * X{problem}(end,:)').^(1/s) / sum(exp(y * X{problem}').^(1/s));
                % Compute w term for max model
                option_utilities = y * X{problem}';
                if (option_utilities(end) == max(option_utilities))
                    w_max{problem}(i) = 1;
                else
                    w_max{problem}(i) = 0;
                end
                                                     
            end
        end
        
        % Create an indicator vector such that
        % z = 1 if x has the highest utility, else z = 0
        z = zeros(nsamples,1);
        for i=1:nsamples
            if (max(priorsamples(i,:)) == priorsamples(i,end))
                z(i) = 1;
            else
                z(i) = 0;
            end
        end
        
        % Compute expected values
        for problem = 1:nproblems
            % 1) E(u_x) -- logit model assuming utility-matching function
            allmeans{problem} = w{problem} * priorsamples ./ sum(w{problem});
            means(problem) = allmeans{problem}(end);
            % 2) E(u_x) --  logit model assuming utility-maximizing function
            allmeans_max{problem} = w_max{problem} * priorsamples ./ sum(w{problem});
            means_max(problem) = allmeans_max{problem}(end);
            % 3) Surprise
            surprise(problem) = 1 / (sum(w{problem}) / nsamples);
            % 4) E(Z), where Z = 1 if x has highest utility, else Z = 0
            pHighest(problem) = w{problem} * z ./ sum(w{problem});
        end
        
    
    end
    
    % Compute the rankings
    
    % 1) Means
    sortedRankingMeans = zeros(1,nproblems);
    [sortedMeans, sortingIndex] = sort(means);
    
    % Loop through and collect a group of items that are not significantly different
    % from one another
    currRank = 1;
    eqClass = [1];
    for p=2:nproblems
        if (abs(sortedMeans(p) - sortedMeans(p-1)) > DIFFERENCE_THRESHOLD)
            % Record the fractional rank of all items in the equivalence class
            fractionalRank = sum(currRank:(currRank+length(eqClass)-1)) / length(eqClass);
            sortedRankingMeans(eqClass) = fractionalRank;
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
    sortedRankingMeans(eqClass) = fractionalRank;
    % Now "invert" the ranking vector so they are ordered by problem number
    rankingMeans(sortingIndex) = sortedRankingMeans;
    
    
    % 2) Means (max)
    sortedRankingMeans_max = zeros(1,nproblems);
    [sortedMeans_max, sortingIndex] = sort(means_max);
    
    % Loop through and collect a group of items that are not significantly different
    % from one another
    currRank = 1;
    eqClass = [1];
    for p=2:nproblems
        if (abs(sortedMeans_max(p) - sortedMeans_max(p-1)) > DIFFERENCE_THRESHOLD)
            % Record the fractional rank of all items in the equivalence class
            fractionalRank = sum(currRank:(currRank+length(eqClass)-1)) / length(eqClass);
            sortedRankingMeans_max(eqClass) = fractionalRank;
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
    sortedRankingMeans_max(eqClass) = fractionalRank;
    % Now "invert" the ranking vector so they are ordered by problem number
    rankingMeans_max(sortingIndex) = sortedRankingMeans_max;
    
    
    % 3) Surprise
    sortedRankingSurprise = zeros(1,nproblems);
    [sortedSurprise, sortingIndex] = sort(surprise);
    
    % Loop through and collect a group of items that are not significantly different
    % from one another
    currRank = 1;
    eqClass = [1];
    for p=2:nproblems
        if (abs(sortedSurprise(p) - sortedSurprise(p-1)) > DIFFERENCE_THRESHOLD)
            % Record the fractional rank of all items in the equivalence class
            fractionalRank = sum(currRank:(currRank+length(eqClass)-1)) / length(eqClass);
            sortedRankingSurprise(eqClass) = fractionalRank;
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
    sortedRankingSurprise(eqClass) = fractionalRank;
    % Now "invert" the ranking vector so they are ordered by problem number
    rankingSurprise(sortingIndex) = sortedRankingSurprise;
    

    
    % 4) P(x_u is highest)
    sortedRankingHighest = zeros(1,nproblems);
    [sortedHighest, sortingIndex] = sort(pHighest);
    
    % Loop through and collect a group of items that are not significantly different
    % from one another
    currRank = 1;
    eqClass = [1];
    for p=2:nproblems
        if (abs(sortedHighest(p) - sortedHighest(p-1)) > DIFFERENCE_THRESHOLD)
            % Record the fractional rank of all items in the equivalence class
            fractionalRank = sum(currRank:(currRank+length(eqClass)-1)) / length(eqClass);
            sortedRankingHighest(eqClass) = fractionalRank;
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
    sortedRankingHighest(eqClass) = fractionalRank;
    % Now "invert" the ranking vector so they are ordered by problem number
    rankingHighest(sortingIndex) = sortedRankingHighest;
    
    
    % Save the results
 
%save('modelpredictions','means','means_max','surprise','labels','rankingMeans','rankingMeans_max','rankingSurprise','pHighest','rankingHighest');

delete(pp);