library(survival)
library(ggplot2)
library(ggfortify)

setwd('../../data/spreadsheets')
learning = read.csv('learning.csv')

# subset data, we only analyze dom and sub demonstrators
learning = learning[learning$STATUS %in% c(0, 1, 2), ]
learning$STATUS = as.factor(learning$STATUS) # type of demonstrator: 0 - None, 1 - dom, 2 - sub

##################################################################################
# check whether dom or sub demonstrators reduce time to group consensus

km_fit <- survfit(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
autoplot(km_fit)

cox = coxph(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
summary(cox)

# relevel status to see all effects
learning$STATUS = relevel(learning$STATUS, 2)

km_fit <- survfit(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
autoplot(km_fit)

cox = coxph(Surv(TRIAL, SURVIVAL) ~ STATUS, data=learning)
summary(cox)

# test assumptions

czph <- cox.zph(cox)
czph
