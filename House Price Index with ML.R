library(tidyverse)
library(corrplot)
library(car)
library(caret)
library(leaps)
library(olsrr)
library(glmnet)

#reading data
us.housing = read_csv(file.choose())
head(us.housing)

#manipulating data
us.housing.by.year= us.housing %>% 
  separate(Date, sep = "-", into = c("Year", "Month", "Day")) %>% 
  mutate_at(c('Year', 'Month', 'Day'), as.numeric) %>%
  select(-Month, -Day)

head(us.housing.by.year)
pairs(us.housing.by.year)

#correlation
cor(us.housing.by.year, use="all.obs", method="pearson") 
corrplot(cor(us.housing.by.year), method = "color", addCoef.col="grey", order = "AOE")

#setting the seed
set.seed(123)
sample.data= sample(c(TRUE, FALSE), nrow(us.housing.by.year), 
                    replace=TRUE, prob=c(0.7,0.3))
train.data  = us.housing.by.year[sample.data, ]
test.data  = us.housing.by.year[!sample.data, ]

#linear model
linear.model = lm(House_Price_Index ~., data = train.data)
print(linear.model)
summary(linear.model)

AIC(linear.model)
BIC(linear.model)
vif(linear.model)
ols_mallows_cp(linear.model, linear.model)

linear.model.prediction = linear.model %>% 
  predict(test.data)

RMSE(linear.model.prediction, test.data$House_Price_Index)
R2(linear.model.prediction, test.data$House_Price_Index)

#stepwise regresion model
variable.elimination = regsubsets(House_Price_Index ~ ., data = train.data, 
                                  nvmax = 8)
results = summary(variable.elimination)

tibble(predictors = 1:8,
       adj_R2 = results$adjr2,
       Cp = results$cp,
       BIC = results$bic) %>%
  gather(statistic, value, -predictors) %>%
  ggplot(aes(predictors, value, color = statistic)) +
  geom_line(show.legend = F) +
  geom_point(show.legend = F) +
  facet_wrap(~ statistic, scales = "free")

coef(variable.elimination, 4)

reduced.linear.model = lm(House_Price_Index ~ Year + Consumer_Price_Index + 
                          Real_GDP + Real_Disposable_Income, 
                          data= train.data)
print(reduced.linear.model)
summary(reduced.linear.model)

AIC(reduced.linear.model)
BIC(reduced.linear.model)
vif(reduced.linear.model)
ols_mallows_cp(reduced.linear.model, linear.model)

reduced.linear.model.prediction = reduced.linear.model %>% 
  predict(test.data)

RMSE(reduced.linear.model.prediction, test.data$House_Price_Index)
R2(reduced.linear.model.prediction, test.data$House_Price_Index)

#ridge regression
x = model.matrix(House_Price_Index~., data = train.data)[,-1]
y = train.data$House_Price_Index

ridge.cross.validation = cv.glmnet(x, y, alpha = 0)
best.lambda = ridge.cross.validation$lambda.min
best.lambda

plot(ridge.cross.validation)

ridge.model = glmnet(x, y, alpha = 0, lambda = best.lambda)
coef(ridge.model)

ridge.test.data = model.matrix(House_Price_Index~., test.data)[,-1]

ridge.model.prediction = ridge.model %>% 
  predict(ridge.test.data)

RMSE(ridge.model.prediction, test.data$House_Price_Index)
R2(ridge.model.prediction, test.data$House_Price_Index)

#lasso regression
lasso.cross.validation = cv.glmnet(x, y, alpha = 1)
best.lambda.lasso = lasso.cross.validation$lambda.min
best.lambda.lasso

plot(lasso.cross.validation)

lasso.model = glmnet(x, y, alpha = 1, lambda = best.lambda.lasso)
coef(lasso.model)

lasso.test.data = model.matrix(House_Price_Index~., test.data)[,-1]

lasso.model.prediction = lasso.model %>% 
  predict(lasso.test.data)

RMSE(lasso.model.prediction, test.data$House_Price_Index)
R2(lasso.model.prediction, test.data$House_Price_Index)

#elastic net 
elastic.net.model = train(House_Price_Index ~., data = train.data, 
                          method = "glmnet", 
                          trControl = trainControl("cv", number = 10), 
                          tuneLength = 10)

plot(elastic.net.model)

best.tune = elastic.net.model$bestTune
best.tune

elastic.test.data = model.matrix(House_Price_Index ~., test.data)[,-1]

elastic.model.prediction = elastic.net.model %>% 
  predict(elastic.test.data)

RMSE(elastic.model.prediction, test.data$House_Price_Index)
R2(elastic.model.prediction, test.data$House_Price_Index)

