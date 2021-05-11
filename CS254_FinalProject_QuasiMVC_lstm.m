
clear all; close all; clc

train = 1; 

if train == 1
   
    % Import Data
    load("LSTM_Input")
    load("LSTM_Output")

    % Remove Col (e.g. subjects) that have zero strides
    [~, c] = size(lstm_input);
    i = 1;
    while i < c
        if isempty(lstm_input{1,i})
            lstm_input(:,i) = []; 
            lstm_output(:,i) = [];
            [~, c] = size(lstm_input);
        else
            i = i+1;
        end 
    end
    
    
    % Split Training, Testing, Validation 
    [~, c] = size(lstm_input);
    rng('default'); s = rng; % For reproducibility
    cols = randperm(c); % Randomly ordered indices 

    num_train = round(.8 * c); 
    num_test = c - num_train; %num_test = .2 * c;
    

    x_train = lstm_input(:,cols(1:num_train));
    y_train = lstm_output(:,cols(1:num_train));

    x_test = lstm_input(:,cols(num_train+1:end));
    y_test = lstm_output(:,cols(num_train+1:c));

    % Split trainig into training + validation 
    [~, c] = size(x_train);
    rng('default'); ss = rng;
    cols_train = randperm(c); % Randomly ordered indices 

    num_train =round(.8 * c); 
    num_val = c-num_train;
    
    x_val = x_train(:,cols_train(num_train+1:end));
    y_val = y_train(:,cols_train(num_train+1:end));

    x_train = x_train(:,cols_train(1:num_train));
    y_train = y_train(:,cols_train(1:num_train));

    %% Normalize Training, Testing, Validation predictors to have zero mean and unit var

    mu = mean([x_train{:}],2);
    sig = std([x_train{:}],0,2);

    [r, c] = size(x_train);

    x_trainNorm = cell(size(x_train));

    for k = 1:c %iterate across columns 

        for j = 1:r % iterate down rows

            if isempty(x_train{j,k})
                continue
            else
                mumat = repmat(mu,[1,length(x_train{j,k})]);
                sigmat = repmat(sig,[1,length(x_train{j,k})]);
                x_trainNorm{j,k} = (x_train{j,k} - mumat) ./ sigmat;
            end

        end
    end

    % Normalze Testing Predictors

    [r, c] = size(x_test);

    x_testNorm = cell(size(x_test));

    for k = 1:c %iterate across columns 

        for j = 1:r % iterate down rows

            if isempty(x_test{j,k})
                continue
            else
                mumat = repmat(mu,[1,length(x_test{j,k})]);
                sigmat = repmat(sig,[1,length(x_test{j,k})]);
                x_testNorm{j,k} = (x_test{j,k} - mumat) ./ sigmat;
            end

        end
    end

    % Normalize Validation Predictors

    [r, c] = size(x_val);

    x_valNorm = cell(size(x_val));

    for k = 1:c %iterate across columns 

        for j = 1:r % iterate down rows

            if isempty(x_val{j,k})
                continue
            else
                mumat = repmat(mu,[1,length(x_val{j,k})]);
                sigmat = repmat(sig,[1,length(x_val{j,k})]);
                x_valNorm{j,k} = (x_val{j,k} - mumat) ./ sigmat;
            end

        end
    end

    %% Prepare for Padding 

    % Sort training data by sequence length

    for i=1:numel(x_trainNorm)
        sequence = x_trainNorm{i};
        sequenceLengths(i) = size(sequence,2);
    end

    [sequenceLengths,idx] = sort(sequenceLengths,'descend');
    x_trainNorm = x_trainNorm(idx);
    y_train = y_train(idx);

    % Delete Empty Cells
    notempt = find(~cellfun(@isempty,x_trainNorm));
    x_trainNorm = x_trainNorm(notempt);
    y_train = y_train(notempt);

    % Make y_train have the same sequence length as the corresponding
    % predictors
    y_trainReshape = cell(size(x_trainNorm));
    for i = 1:length(x_trainNorm)
        y_trainReshape{i} = repmat(y_train{i},1,length(x_trainNorm{i}));
    end

    % y_test
    x_testNorm = reshape(x_testNorm.',1,[]); % flatten
    notempt = find(~cellfun(@isempty, x_testNorm)); % remove empty cells 
    x_testNorm = x_testNorm(notempt);
    y_test = reshape(y_test.',1,[]); % flatten
    y_test = y_test(notempt);

    y_testReshape = cell(size(x_testNorm));
    for i = 1:length(x_testNorm)
        y_testReshape{i} = repmat(y_test{i},1,length(x_testNorm{i}));
    end

    % y_val
    x_valNorm = reshape(x_valNorm.',1,[]); % flatten
    notempt = find(~cellfun(@isempty, x_valNorm)); % remove empty cells 
    x_valNorm = x_valNorm(notempt);
    y_val = reshape(y_val.',1,[]); % flatten y val
    y_val = y_val(notempt);

    y_valReshape = cell(size(x_valNorm));
    for i = 1:length(x_valNorm)
        y_valReshape{i} = repmat(y_val{i},[1,length(x_valNorm{i})]);
    end


    %% Define Network Architecture 

    numFeatures = size(x_trainNorm{1},1);
    numResponses = size(y_train{1},1);
    numHiddenUnits = 5;

    layers = [ ...
        sequenceInputLayer(numFeatures)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        fullyConnectedLayer(4)
        dropoutLayer(0.2)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    

    maxEpochs = 15;
    %miniBatchSize = 245;  %With full subject set: 735/35 - 21 equal parts   % 475/25 = 19 equal parts 
    miniBatchSize = 105;
    
    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'InitialLearnRate',0.005, ...
        'GradientThreshold',1, ...
        'Shuffle','never', ...
        'Plots','training-progress',...
        'Verbose',0,...
        'ValidationData',{x_valNorm,y_valReshape});
    
    
    %% Train Network


    net = trainNetwork(x_trainNorm,y_trainReshape,layers,options);

    LSTMnet12 = net;
    save LSTMnet12

else
  
    %% Test 

    net = load('LSTMnet12');
    
    YPred = predict(net,x_valNorm,'MiniBatchSize',1);
    
end  


%% Analysis


% Compute single Quasi-MVC values from strides ///////////////////


% Last Val 
% Get the last estimate from each stride
for i = 1:numel(y_valReshape)
    y_valReshapeLast(i) = y_valReshape{i}(end);
    YPredLast(i) = YPred{i}(end);
end

figure
rmse = sqrt(mean((abs(YPredLast - y_valReshapeLast)).^2));
histogram(abs(YPredLast - y_valReshapeLast))
title("Last Value RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")

% First Val 
% Get the first estimate from each stride
for i = 1:numel(y_valReshape)
    y_valReshapeFirst(i) = y_valReshape{i}(1);
    YPredFirst(i) = YPred{i}(1);
end

figure
rmse = sqrt(mean((abs(YPredFirst - y_valReshapeFirst)).^2))
histogram(abs(YPredFirst - y_valReshapeFirst))
title("First Value RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")


% Mean Val 
% Get the mean estimate from each stride
for i = 1:numel(y_valReshape)
    y_valReshapeMean(i) = mean(y_valReshape{i});
    YPredMean(i) = mean(YPred{i});
end

figure
rmse = sqrt(mean((abs(YPredMean - y_valReshapeMean)).^2))
histogram(abs(YPredMean - y_valReshapeMean))
title("Mean Value RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")

% Median Val 
% Get the median estimate from each stride
for i = 1:numel(y_valReshape)
    y_valReshapeMedian(i) = median(y_valReshape{i});
    YPredMedian(i) = median(YPred{i});
end

figure
rmse = sqrt(mean((abs(YPredMedian - y_valReshapeMedian)).^2))
histogram(abs(YPredMedian - y_valReshapeMedian))
title("Median Value RMSE = " + rmse)
ylabel("Frequency")
xlabel("Error")
    
    

% //////////////////////
% Get a Quasi-MVC estimate for each subject (not by stride, but by subject) 
    
%     First sub: 46 strides
%     2nd: 35  % lowest number of strides
%     3rd: 36
%     4th: 41
%     5th: 40
    
    
% for a fair stride count comparison, go until 35 strides
y_valshort = y_val(1:175);
y_val_unshape = reshape(y_valshort.',5,35); % unreshape such that each row is a subject


% get mean ypred from each stride since mean resulted in best MVC prediction
for i = 1:length(x_valNorm)
    meanYpred(i) = mean(mean(YPred{i}));
end

% get short verion of median Ypred
yPredshort = meanYpred(1:175);
% Unshape ypredshort
yPred_unshape = reshape(yPredshort.',5,35);

% Get short version of x_valNorm
x_valshort = x_valNorm(1:175);
% get median value in each cell
for i = 1:175
    temp = x_valshort{i};
    xval_mean(i) = mean(temp(4,:));
end
% Upshape xvalmed
xval_unshape = reshape(xval_mean.',5,35);


% get the mean value from each subject for yval and ypred
MVC = mean(cell2mat(y_val_unshape),2);
QuasiMVC = mean(yPred_unshape,2);
WalksEMG = mean(xval_unshape,2);

% Plot mean stride EMG vs MVC for true and predicted (quasi)
figure
scatter(WalksEMG, QuasiMVC,70, 'filled', 'MarkerFaceColor',[0 0.4470 0.7410])
hold on
scatter(WalksEMG, MVC, 70, 'd', 'filled', 'MarkerFaceColor',[0.4660 0.6740 0.1880])
xlabel('Mean Stride EMG (mv)')
ylabel('Mean MVC (mv)')
legend('Quasi-MVC', 'True MVC')
set(gca,'box','off','tickdir','out','fontsize',16)




