library(lmtest)
library(lme4)
library(boot)

setwd('../../data/spreadsheets')
social_parameters = read.csv('social_parameters.csv')
social_parameters$social_status = relevel(social_parameters$social_status, 'SUB')

boxplot(logit(noise_frequency) ~ social_status, data=social_parameters)

model <- lm(logit(noise_frequency) ~ social_status, data=social_parameters)
plot(model)
shapiro.test(model$residuals) # W = 0.96403, p-value = 0.07423
bptest(model) # BP = 2.8357, df = 1, p-value = 0.09219 (BP is chi²)

summary(model)

