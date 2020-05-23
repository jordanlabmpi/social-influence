library(lmtest)
library(lme4)
library(boot)

setwd('../../data/spreadsheets')
social_parameters = read.csv('social_parameters.csv')
social_parameters$social_status = as.factor(social_parameters$social_status)
social_parameters$social_status = relevel(social_parameters$social_status, 'SUB')

boxplot(noise_frequency ~ social_status, data=social_parameters)

model <- lm(noise_frequency ~ social_status, data=social_parameters)
plot(model)
shapiro.test(model$residuals) # W = 0.96403, p-value = 0.07423
bptest(model) # BP = 2.8357, df = 1, p-value = 0.09219 (BP is chi²)

summary(model)

aggregate(social_parameters$noise_frequency, FUN = mean, by = list(social_parameters$social_status))
