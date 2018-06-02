function [missClassification] = discrim(w,gamm,test, features)


    XTest = test(:, features+1);
    yTest = test(:, 1);
    predict = XTest*w - gamm > 0;
    missClassification = sum(predict~=yTest);