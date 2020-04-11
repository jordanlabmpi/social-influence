library(survival)
library(ggplot2)
library(ggfortify)

setwd('~/Documents/BurtoniLearningExp/learning/')
learning = read.csv('LearningResults.csv')

# subset data, we only analyze dom and sub demonstrators
learning = learning[learning$STATUS %in% c(0, 1, 2), ]
learning$DEMONSTRATOR = as.factor(learning$DEMONSTRATOR) # has demonstrator: 1 - yes or 0 - no
learning$STATUS = as.factor(learning$STATUS) # type of demonstrator: 0 - None, 1 - dom, 2 - sub

##################################################################################
# check whether dom or sub demonstrators reduce time to group consensus

km_fit <- survfit(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
autoplot(km_fit)

cox = coxph(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
summary(cox)

# switch dom and None factors to see all effects

learning = read.csv('LearningResults.csv')
learning = learning[learning$STATUS %in% c(0, 1, 2), ]
learning$STATUS[learning$STATUS == 0] = 3
learning$STATUS[learning$STATUS == 1] = 0
learning$STATUS[learning$STATUS == 3] = 1
learning$STATUS = as.factor(learning$STATUS)

km_fit <- survfit(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
autoplot(km_fit)

cox = coxph(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
summary(cox)

# test assumptions

czph <- cox.zph(cox)
czph
