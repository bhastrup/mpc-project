data {
  int<lower=0>  N_slots;              // number of ad slots
  vector[N_slots] cost;               // outcome vector
  vector[N_slots] u;                  // predictor vector
  vector<lower=0>[N_slots] weights;   // model weights
}

parameters {
  real alpha;                   // intercept
  vector[N_slots] beta;         // coefficients for predictors
  real<lower=0> sigma;          // error sd
}

model {
  real cost_pred[N_slots];

  beta ~ normal(0,1);

  for(i in 1:N_slots){
    cost_pred[i] = a + u * b;
  }
  for(i in 1:N_slots){
    target += normal_lpdf(cost[i] | cost_pred[i], sigma) * weights[i];
  }
}
