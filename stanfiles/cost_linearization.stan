data {
  int<lower=0>  N_slots;              // number of ad slots
  vector[N_slots] cost;               // outcome vector
  vector[N_slots] u;                  // predictor vector
  vector<lower=0>[N_slots] weights;   // model weights
}

parameters {
  real a;                    // intercept
  real b;                    // coefficients for predictors
  real<lower=0> sigma;       // error sd
}

model {
    cost ~ normal(a + u * b, sigma);

    for (n in 1:N_slots) {
        target += weights[N_slots] * normal_lpdf(cost[N_slots] | a*u[N_slots] + b, sigma);
}
}
