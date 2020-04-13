library(permute)
library(MASS)
library(ggplot2)
library(patchwork)

# this script is an almost identical implementation of the python code, but much slower
# the only difference is in the calculation of the p-value, the estimated distribution should be non-parametric pdf
# for the paper, we used the python code and visualizations

#### define functions that calculate ####

# per trial difference of a metric between doms and subs (calculate_diff_per_trial)

calculate_diff_per_trial = function (social_parameters, n_trials, metric, intercept = FALSE) {
  aggregated = aggregate(social_parameters[, metric] ~ social_parameters$trial + social_parameters$social_status, FUN = mean)
  aggregated = setNames(aggregated, c('trial', 'social_status', metric))
  differences = vector("numeric", length=n_trials)
  intercepts = vector("numeric", length=n_trials)
  for (trial in seq(n_trials) - 1) {
    intercepts[trial + 1] = aggregated[, metric][aggregated$trial == trial & aggregated$social_status == 'SUB']
    differences[trial + 1] = aggregated[, metric][aggregated$trial == trial & aggregated$social_status == 'DOM'] - intercepts[trial + 1]
  }
  estimate = mean(differences)
  intercept = mean(intercepts)
  if (intercept) {
    return(c(estimate, intercept))
  } else {
    return(estimate)
  }
}

# a two-sided p-value based on estimated distribution and observed estimate (calculate_p_value)

calculate_p_value = function (observed, estimates) {
  distribution = fitdistr(estimates$x, densfun = 'normal') # this should be non-parametric (pdf) instead, as implemented in pyhton code
  if (observed > distribution$estimate['mean']) {
    p_value = pnorm(observed, distribution$estimate['mean'], sd = distribution$estimate['sd'], lower.tail = FALSE)
  } else {
    p_value = pnorm(observed, distribution$estimate['mean'], sd = distribution$estimate['sd'])
  }
  return(2 * p_value)
}

setwd('../../data/spreadsheets/')

n_shuffles = 1000
n_trials = 6
metric = 'centrality' # pairwise_dist, aa_out, centrality (noise_frequency is not a 'network parameter' because only derived from one node's actions)

social_parameters = read.csv('social_parameters.csv')

## calculate observed difference
diff_intercept = calculate_diff_per_trial(social_parameters, n_trials, metric, intercept = TRUE)
observed = diff_intercept[1]
intercept = diff_intercept[2]

## randomized differences
estimates = vector("numeric", length=n_shuffles)
intercepts = vector("numeric", length=n_shuffles)
estimates_all = vector("numeric", length=n_shuffles)
intercepts_all = vector("numeric", length=n_shuffles)
for (idx in seq(n_shuffles)) {
  ## perform network (node) randomization
  social_parameters$social_status = social_parameters$social_status[shuffle(social_parameters$X, control = how(blocks = social_parameters$trial))]
  diff_intercept = calculate_diff_per_trial(social_parameters, n_trials, metric, intercept = TRUE)
  estimates[idx] = diff_intercept[1]
  intercepts[idx] = diff_intercept[2]
}
estimates = data.frame(x = estimates)
intercepts = data.frame(x = intercepts)

## visualize

segment_data = data.frame(
  x = rep(1, length(estimates$x)),
  xend = rep(2, length(estimates$x)),
  y = intercepts$x,
  yend = intercepts$x + estimates$x
)

p1 = ggplot() +
  geom_segment(data = segment_data, aes(x = x, y = y, xend = xend, yend = yend), alpha = 0.1) +
  geom_vline(xintercept = 1.1, 
             color = 'gray', lwd = 0.5, linetype = 'dashed') +
  geom_segment(aes(x = 1, y = intercept, xend = 2, yend = observed + intercept), col = 'red') +
  theme_classic() +
  xlab('social_status') + 
  xlim('sub', 'dom') +
  ylab(metric)

distribution = fitdistr(estimates$x, densfun = 'normal')
p2 = ggplot(data = estimates) +
  geom_histogram(aes(x = x, y = ..density..), col = 'black', fill = 'white', bins = 40) +
  stat_function(fun = dnorm, 
                args = list(mean = distribution$estimate['mean'], sd = distribution$estimate['sd'], log = FALSE), 
                color='black', lwd = 0.5, linetype = 'dashed') +
  geom_vline(xintercept = observed, 
             color = 'red', lwd = 0.5) +
  theme_classic() +
  xlab('effect_size')

p1 | p2

## the p-value
calculate_p_value(observed, estimates)

