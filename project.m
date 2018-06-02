clear;

dataFile = 'wdbc.data';
features = 30;  % size of a feature vector
divider = repmat('=', [1, 75]);


%% Q1

mu = 0.0001; 
fracTest = 0.20;
reord = 0;  
[train,~,~,~] = wdbcData(dataFile,features,fracTest,reord);
[w, gamm, obj, misclass] = separateQP(train, 1:features, mu);

disp(divider);
fprintf('mu = %f\nfracTest = %f\nreord = %i\n', mu, fracTest, reord);
disp('w = ');
disp(w');
fprintf('gamma = %f\n', gamm);
fprintf('optimal objective value for QP = %f\n', obj);
fprintf('misclassifications for QP = %i\n', misclass);

%% Q2

tau = 0.001; 
fracTest = 0.20;
reord = 0;
[train,~,~,~] = wdbcData(dataFile, features, fracTest, reord);
[w, gamm, obj, misclass] = separateL1(train, 1:features, tau);

for i = 1:length(w)
if isequal(abs(w(i)) < 10^(-6)*norm(w, Inf), 1)
        w(i) = 0;
end
end

disp(divider);
fprintf('mu = %f\nfracTest = %f\nreord = %i\n', tau, fracTest, reord);
disp('w = ');
disp(w');
fprintf('gamma = %f\n', gamm);
fprintf('optimal objective value for L1 = %f\n', obj);
fprintf('misclassifcations for L1 = %i\n', misclass);


%% Q3 QP
mu = 0.001;  % regularization term

fracTests = [0.1 0.15 0.2 0.05];
reords = [0 0 1 1];
for testNum = 1 : length(fracTests)
    fracTest = fracTests(testNum);
    reord = reords(testNum);
    
    [train,test,~,~] = wdbcData(dataFile,features,fracTest,reord);
    [w, gamm, ~, ~] = separateQP(train, 1:features, mu);
    [missClassification] = discrim(w,gamm,test,1:features);
    
    
    disp(divider);
    fprintf('fracTest = %f\nreord = %i\n', fracTest, reord);
    fprintf('misclassifications for QP = %i\n', missClassification);
end

%% Q3 L1
%LP
tau = 0.1;  % regularization term

fracTests = [0.1 0.15 0.2 0.05];
reords = [0 0 1 1];
for testNum = 1 : length(fracTests)
    fracTest = fracTests(testNum);
    reord = reords(testNum);
    
    [train,test,~,~] = wdbcData(dataFile,features,fracTest,reord);
    [w, gamm, ~, ~] = separateL1(train, 1:features, tau);
    [missClassification] = discrim(w,gamm,test,1:features);
    
    disp(divider);
    fprintf('fracTest=%f\nreord=%i\n', fracTest, reord);
    fprintf('misclassifications for L1 = %i\n', missClassification);
end

%% Q4
tau = 0.1;
fracTest = 0.1;
reord = 0;

[train,test,ntrain,ntest] = wdbcData(dataFile,features,fracTest,reord);
[w, ~, obj, misclass] = separateL1(train, 1:features, tau);


for i = 1:length(w)
if isequal(abs(w(i)) < 10^(-6)*norm(w, Inf), 1)
        w(i) = 0;
end
end

while sum(w~=0)~= 2
    tau = tau*2;
    

    [train,test,ntrain,ntest] = wdbcData(dataFile,features,fracTest,reord);
    [w, ~, obj, misclass] = separateL1(train, 1:features, tau);

    for i = 1:length(w)
        if isequal(abs(w(i)) < 10^(-6)*norm(w, Inf), 1)
        w(i) = 0;
        end
    end
end

bestFeatures = find(w~=0);

%run QP using only best features
mu = 0.0001; 
[w, gamm, ~, misclassTrain] = separateQP(train, bestFeatures, mu);
[missClassification] = discrim(w,gamm,test,bestFeatures);

disp(divider);
fprintf('bestFeatures = %d, %d\n', bestFeatures);
fprintf('w = %f, %f\n', w);
fprintf('misclassification = %i\n', missClassification);

figure; hold on;

% perturb the points to prevent coalesence
% + plots
XTest = test(:, bestFeatures+1);
yTest = test(:, 1);
x1 = rand(size(XTest(yTest==1, 1)))*(.05) + XTest(yTest==1, 1);
y1 = rand(size(XTest(yTest==1, 2)))*(.05) + XTest(yTest==1, 2);

% o plots
x2 = rand(size(XTest(yTest==0, 1)))*(.05) + XTest(yTest==0, 1);
y2 = rand(size(XTest(yTest==0, 2)))*(.05) + XTest(yTest==0, 2);

%plots the points 
plot(x1, y1, '+');
plot(x2, y2, 'o');
slope = -w(1)/w(2);
intercept = gamm/ w(2);
refline(slope, intercept);
xlabel(sprintf('Feature %i', bestFeatures(1)));
ylabel(sprintf('Feature %i', bestFeatures(2)));
legend('Malignant', 'Benign');